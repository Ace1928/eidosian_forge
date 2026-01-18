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
class SenderMixin:
    """
    Utility class for sending emails easily.

    Use with SMTPSenderFactory or ESMTPSenderFactory.
    """
    done = 0

    def getMailFrom(self):
        if not self.done:
            self.done = 1
            return str(self.factory.fromEmail)
        else:
            return None

    def getMailTo(self):
        return self.factory.toEmail

    def getMailData(self):
        return self.factory.file

    def sendError(self, exc):
        SMTPClient.sendError(self, exc)
        if self.factory.retries >= 0 or (not exc.retry and (not (exc.code >= 400 and exc.code < 500))):
            self.factory.sendFinished = True
            self.factory.result.errback(exc)

    def sentMail(self, code, resp, numOk, addresses, log):
        self.factory.sendFinished = True
        if code not in SUCCESS:
            errlog = []
            for addr, acode, aresp in addresses:
                if acode not in SUCCESS:
                    errlog.append(addr + b': ' + networkString('%03d' % (acode,)) + b' ' + aresp)
            errlog.append(log.str())
            exc = SMTPDeliveryError(code, resp, b'\n'.join(errlog), addresses)
            self.factory.result.errback(exc)
        else:
            self.factory.result.callback((numOk, addresses))