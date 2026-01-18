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
def _processConnectionError(self, connector, err):
    self.currentProtocol = None
    if self.retries < 0 and (not self.sendFinished):
        log.msg('SMTP Client retrying server. Retry: %s' % -self.retries)
        self.file.seek(0, 0)
        connector.connect()
        self.retries += 1
    elif not self.sendFinished:
        if err.check(error.ConnectionDone):
            err.value = SMTPConnectError(-1, 'Unable to connect to server.')
        self.result.errback(err.value)