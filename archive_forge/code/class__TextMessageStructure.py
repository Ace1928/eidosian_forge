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
class _TextMessageStructure(_SinglepartMessageStructure):
    """
    L{_TextMessageStructure} represents the message structure of a I{text/*}
    message.
    """

    def encode(self, extended):
        """
        A body type of type TEXT contains, immediately after the basic
        fields, the size of the body in text lines.  Note that this
        size is the size in its content transfer encoding and not the
        resulting size after any decoding.
        """
        result = _SinglepartMessageStructure._basicFields(self)
        result.append(getLineCount(self.message))
        if extended:
            result.extend(self._extended())
        return result