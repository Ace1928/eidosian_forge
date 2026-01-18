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
def _authResponse(self, auth, challenge):
    self._failresponse = self.esmtpAUTHDeclined
    try:
        challenge = base64.b64decode(challenge)
    except binascii.Error:
        self.sendLine(b'*')
        self._okresponse = self.esmtpAUTHMalformedChallenge
        self._failresponse = self.esmtpAUTHMalformedChallenge
    else:
        resp = auth.challengeResponse(self.secret, challenge)
        self._expected = [235, 334]
        self._okresponse = self.smtpState_maybeAuthenticated
        self.sendLine(base64.b64encode(resp))