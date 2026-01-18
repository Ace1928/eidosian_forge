import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def __cbFetch(self, results, tag, query, uid):
    if self.blocked is None:
        self.blocked = []
    try:
        id, msg = next(results)
    except StopIteration:
        self.setTimeout(self._oldTimeout)
        del self._oldTimeout
        self.sendPositiveResponse(tag, b'FETCH completed')
        self._unblock()
    else:
        self.spewMessage(id, msg, query, uid).addCallback(lambda _: self.__cbFetch(results, tag, query, uid)).addErrback(self.__ebSpewMessage)