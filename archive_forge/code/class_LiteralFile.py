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
class LiteralFile:
    _memoryFileLimit = 1024 * 1024 * 10

    def __init__(self, size, defered):
        self.size = size
        self.defer = defered
        if size > self._memoryFileLimit:
            self.data = tempfile.TemporaryFile()
        else:
            self.data = BytesIO()

    def write(self, data):
        self.size -= len(data)
        passon = None
        if self.size > 0:
            self.data.write(data)
        else:
            if self.size:
                data, passon = (data[:self.size], data[self.size:])
            else:
                passon = b''
            if data:
                self.data.write(data)
        return passon

    def callback(self, line):
        """
        Call deferred with data and rest of line
        """
        self.data.seek(0, 0)
        self.defer.callback((self.data, line))