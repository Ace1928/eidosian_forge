from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
class _GenericHTTPChannelProtocol(proxyForInterface(IProtocol, '_channel')):
    """
    A proxy object that wraps one of the HTTP protocol objects, and switches
    between them depending on TLS negotiated protocol.

    @ivar _negotiatedProtocol: The protocol negotiated with ALPN or NPN, if
        any.
    @type _negotiatedProtocol: Either a bytestring containing the ALPN token
        for the negotiated protocol, or L{None} if no protocol has yet been
        negotiated.

    @ivar _channel: The object capable of behaving like a L{HTTPChannel} that
        is backing this object. By default this is a L{HTTPChannel}, but if a
        HTTP protocol upgrade takes place this may be a different channel
        object. Must implement L{IProtocol}.
    @type _channel: L{HTTPChannel}

    @ivar _requestFactory: A callable to use to build L{IRequest} objects.
    @type _requestFactory: L{IRequest}

    @ivar _site: A reference to the creating L{twisted.web.server.Site} object.
    @type _site: L{twisted.web.server.Site}

    @ivar _factory: A reference to the creating L{HTTPFactory} object.
    @type _factory: L{HTTPFactory}

    @ivar _timeOut: A timeout value to pass to the backing channel.
    @type _timeOut: L{int} or L{None}

    @ivar _callLater: A value for the C{callLater} callback.
    @type _callLater: L{callable}
    """
    _negotiatedProtocol = None
    _requestFactory = Request
    _factory = None
    _site = None
    _timeOut = None
    _callLater = None

    @property
    def factory(self):
        """
        @see: L{_genericHTTPChannelProtocolFactory}
        """
        return self._channel.factory

    @factory.setter
    def factory(self, value):
        self._factory = value
        self._channel.factory = value

    @property
    def requestFactory(self):
        """
        A callable to use to build L{IRequest} objects.

        Retries the object from the current backing channel.
        """
        return self._channel.requestFactory

    @requestFactory.setter
    def requestFactory(self, value):
        """
        A callable to use to build L{IRequest} objects.

        Sets the object on the backing channel and also stores the value for
        propagation to any new channel.

        @param value: The new callable to use.
        @type value: A L{callable} returning L{IRequest}
        """
        self._requestFactory = value
        self._channel.requestFactory = value

    @property
    def site(self):
        """
        A reference to the creating L{twisted.web.server.Site} object.

        Returns the site object from the backing channel.
        """
        return self._channel.site

    @site.setter
    def site(self, value):
        """
        A reference to the creating L{twisted.web.server.Site} object.

        Sets the object on the backing channel and also stores the value for
        propagation to any new channel.

        @param value: The L{twisted.web.server.Site} object to set.
        @type value: L{twisted.web.server.Site}
        """
        self._site = value
        self._channel.site = value

    @property
    def timeOut(self):
        """
        The idle timeout for the backing channel.
        """
        return self._channel.timeOut

    @timeOut.setter
    def timeOut(self, value):
        """
        The idle timeout for the backing channel.

        Sets the idle timeout on both the backing channel and stores it for
        propagation to any new backing channel.

        @param value: The timeout to set.
        @type value: L{int} or L{float}
        """
        self._timeOut = value
        self._channel.timeOut = value

    @property
    def callLater(self):
        """
        A value for the C{callLater} callback. This callback is used by the
        L{twisted.protocols.policies.TimeoutMixin} to handle timeouts.
        """
        return self._channel.callLater

    @callLater.setter
    def callLater(self, value):
        """
        Sets the value for the C{callLater} callback. This callback is used by
        the L{twisted.protocols.policies.TimeoutMixin} to handle timeouts.

        @param value: The new callback to use.
        @type value: L{callable}
        """
        self._callLater = value
        self._channel.callLater = value

    def dataReceived(self, data):
        """
        An override of L{IProtocol.dataReceived} that checks what protocol we're
        using.
        """
        if self._negotiatedProtocol is None:
            try:
                negotiatedProtocol = self._channel.transport.negotiatedProtocol
            except AttributeError:
                negotiatedProtocol = b'http/1.1'
            if negotiatedProtocol is None:
                negotiatedProtocol = b'http/1.1'
            if negotiatedProtocol == b'h2':
                if not H2_ENABLED:
                    raise ValueError('Negotiated HTTP/2 without support.')
                networkProducer = self._channel._networkProducer
                networkProducer.unregisterProducer()
                self._channel.setTimeout(None)
                transport = self._channel.transport
                self._channel = H2Connection()
                self._channel.requestFactory = self._requestFactory
                self._channel.site = self._site
                self._channel.factory = self._factory
                self._channel.timeOut = self._timeOut
                self._channel.callLater = self._callLater
                self._channel.makeConnection(transport)
                networkProducer.registerProducer(self._channel, True)
            else:
                assert negotiatedProtocol == b'http/1.1', 'Unsupported protocol negotiated'
            self._negotiatedProtocol = negotiatedProtocol
        return self._channel.dataReceived(data)