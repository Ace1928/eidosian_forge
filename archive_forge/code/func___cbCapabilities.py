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
def __cbCapabilities(self, result):
    lines, tagline = result
    caps = {}
    for rest in lines:
        for cap in rest[1:]:
            parts = cap.split(b'=', 1)
            if len(parts) == 1:
                category, value = (parts[0], None)
            else:
                category, value = parts
            caps.setdefault(category, []).append(value)
    for category in caps:
        if caps[category] == [None]:
            caps[category] = None
    self._capCache = caps
    return caps