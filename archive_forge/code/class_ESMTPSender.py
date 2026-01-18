import base64
import binascii
import os
import random
import re
import socket
import time
import warnings
from email.utils import parseaddr
from io import BytesIO
from typing import Type
from zope.interface import implementer
from twisted import cred
from twisted.copyright import longversion
from twisted.internet import defer, error, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.interfaces import ISSLTransport, ITLSTransport
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.python.runtime import platform
import codecs
class ESMTPSender(SenderMixin, ESMTPClient):
    requireAuthentication = True
    requireTransportSecurity = True

    def __init__(self, username, secret, contextFactory=None, *args, **kw):
        self.heloFallback = 0
        self.username = username
        self._hostname = kw.pop('hostname', None)
        if contextFactory is None:
            contextFactory = self._getContextFactory()
        ESMTPClient.__init__(self, secret, contextFactory, *args, **kw)
        self._registerAuthenticators()

    def _registerAuthenticators(self):
        self.registerAuthenticator(CramMD5ClientAuthenticator(self.username))
        self.registerAuthenticator(LOGINAuthenticator(self.username))
        self.registerAuthenticator(PLAINAuthenticator(self.username))

    def _getContextFactory(self):
        if self.context is not None:
            return self.context
        if self._hostname is None:
            return None
        try:
            from twisted.internet.ssl import optionsForClientTLS
        except ImportError:
            return None
        else:
            context = optionsForClientTLS(self._hostname)
            return context