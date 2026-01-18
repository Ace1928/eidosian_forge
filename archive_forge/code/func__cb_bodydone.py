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
def _cb_bodydone(self, result, request, url):
    headers = self._headers_from_twisted_response(result['txresponse'])
    respcls = responsetypes.from_args(headers=headers, url=url, body=result['body'])
    try:
        version = result['txresponse'].version
        protocol = f'{to_unicode(version[0])}/{version[1]}.{version[2]}'
    except (AttributeError, TypeError, IndexError):
        protocol = None
    response = respcls(url=url, status=int(result['txresponse'].code), headers=headers, body=result['body'], flags=result['flags'], certificate=result['certificate'], ip_address=result['ip_address'], protocol=protocol)
    if result.get('failure'):
        result['failure'].value.response = response
        return result['failure']
    return response