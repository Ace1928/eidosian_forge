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
def __cbList(self, result, command):
    lines, last = result
    results = []
    for parts in lines:
        if len(parts) == 4 and parts[0] == command:
            parts[1] = tuple((nativeString(flag) for flag in parts[1]))
            parts[2] = parts[2].decode('imap4-utf-7')
            parts[3] = parts[3].decode('imap4-utf-7')
            results.append(tuple(parts[1:]))
    return results