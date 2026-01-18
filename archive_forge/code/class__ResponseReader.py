import ipaddress
import logging
import re
from contextlib import suppress
from io import BytesIO
from time import time
from urllib.parse import urldefrag, urlunparse
from twisted.internet import defer, protocol, ssl
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.internet.error import TimeoutError
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.http import PotentialDataLoss, _DataLoss
from twisted.web.http_headers import Headers as TxHeaders
from twisted.web.iweb import UNKNOWN_LENGTH, IBodyProducer
from zope.interface import implementer
from scrapy import signals
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.exceptions import StopDownload
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.python import to_bytes, to_unicode
class _ResponseReader(protocol.Protocol):

    def __init__(self, finished, txresponse, request, maxsize, warnsize, fail_on_dataloss, crawler):
        self._finished = finished
        self._txresponse = txresponse
        self._request = request
        self._bodybuf = BytesIO()
        self._maxsize = maxsize
        self._warnsize = warnsize
        self._fail_on_dataloss = fail_on_dataloss
        self._fail_on_dataloss_warned = False
        self._reached_warnsize = False
        self._bytes_received = 0
        self._certificate = None
        self._ip_address = None
        self._crawler = crawler

    def _finish_response(self, flags=None, failure=None):
        self._finished.callback({'txresponse': self._txresponse, 'body': self._bodybuf.getvalue(), 'flags': flags, 'certificate': self._certificate, 'ip_address': self._ip_address, 'failure': failure})

    def connectionMade(self):
        if self._certificate is None:
            with suppress(AttributeError):
                self._certificate = ssl.Certificate(self.transport._producer.getPeerCertificate())
        if self._ip_address is None:
            self._ip_address = ipaddress.ip_address(self.transport._producer.getPeer().host)

    def dataReceived(self, bodyBytes):
        if self._finished.called:
            return
        self._bodybuf.write(bodyBytes)
        self._bytes_received += len(bodyBytes)
        bytes_received_result = self._crawler.signals.send_catch_log(signal=signals.bytes_received, data=bodyBytes, request=self._request, spider=self._crawler.spider)
        for handler, result in bytes_received_result:
            if isinstance(result, Failure) and isinstance(result.value, StopDownload):
                logger.debug('Download stopped for %(request)s from signal handler %(handler)s', {'request': self._request, 'handler': handler.__qualname__})
                self.transport.stopProducing()
                self.transport.loseConnection()
                failure = result if result.value.fail else None
                self._finish_response(flags=['download_stopped'], failure=failure)
        if self._maxsize and self._bytes_received > self._maxsize:
            logger.warning('Received (%(bytes)s) bytes larger than download max size (%(maxsize)s) in request %(request)s.', {'bytes': self._bytes_received, 'maxsize': self._maxsize, 'request': self._request})
            self._bodybuf.truncate(0)
            self._finished.cancel()
        if self._warnsize and self._bytes_received > self._warnsize and (not self._reached_warnsize):
            self._reached_warnsize = True
            logger.warning('Received more bytes than download warn size (%(warnsize)s) in request %(request)s.', {'warnsize': self._warnsize, 'request': self._request})

    def connectionLost(self, reason):
        if self._finished.called:
            return
        if reason.check(ResponseDone):
            self._finish_response()
            return
        if reason.check(PotentialDataLoss):
            self._finish_response(flags=['partial'])
            return
        if reason.check(ResponseFailed) and any((r.check(_DataLoss) for r in reason.value.reasons)):
            if not self._fail_on_dataloss:
                self._finish_response(flags=['dataloss'])
                return
            if not self._fail_on_dataloss_warned:
                logger.warning("Got data loss in %s. If you want to process broken responses set the setting DOWNLOAD_FAIL_ON_DATALOSS = False -- This message won't be shown in further requests", self._txresponse.request.absoluteURI.decode())
                self._fail_on_dataloss_warned = True
        self._finished.errback(reason)