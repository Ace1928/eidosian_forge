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
def _ebAuthenticated(self, reason):
    """
        Handle cred login errors by translating them to the SMTP authenticate
        failed.  Translate all other errors into a generic SMTP error code and
        log the failure for inspection.  Stop all errors from propagating.

        @param reason: Reason for failure.
        """
    self.challenge = None
    if reason.check(cred.error.UnauthorizedLogin):
        self.sendCode(535, b'Authentication failed')
    else:
        log.err(reason, 'SMTP authentication failure')
        self.sendCode(451, b'Requested action aborted: local error in processing')