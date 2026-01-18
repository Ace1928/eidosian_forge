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
def __cbStatus(self, result):
    lines, last = result
    status = {}
    for parts in lines:
        if parts[0] == b'STATUS':
            items = parts[2]
            items = [items[i:i + 2] for i in range(0, len(items), 2)]
            for k, v in items:
                try:
                    status[nativeString(k)] = v
                except UnicodeDecodeError:
                    raise IllegalServerResponse(repr(items))
    for k in status.keys():
        t = self.STATUS_TRANSFORMATIONS.get(k)
        if t:
            try:
                status[k] = t(status[k])
            except Exception as e:
                raise IllegalServerResponse('(' + k + ' ' + status[k] + '): ' + str(e))
    return status