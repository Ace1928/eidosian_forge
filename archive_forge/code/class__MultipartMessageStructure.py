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
class _MultipartMessageStructure(_MessageStructure):
    """
    L{_MultipartMessageStructure} represents the message structure of a
    I{multipart/*} message.
    """

    def __init__(self, message, subtype, attrs):
        """
        @param message: An L{IMessagePart} provider which this structure object
            reports on.

        @param subtype: A L{str} giving the MIME subtype of the message (for
            example, C{"plain"}).

        @param attrs: A C{dict} giving the parameters of the I{Content-Type}
            header of the message.
        """
        _MessageStructure.__init__(self, message, attrs)
        self.subtype = subtype

    def _getParts(self):
        """
        Return an iterator over all of the sub-messages of this message.
        """
        i = 0
        while True:
            try:
                part = self.message.getSubPart(i)
            except IndexError:
                break
            else:
                yield part
                i += 1

    def encode(self, extended):
        """
        Encode each sub-message and added the additional I{multipart} fields.
        """
        result = [_getMessageStructure(p).encode(extended) for p in self._getParts()]
        result.append(self.subtype)
        if extended:
            result.extend(self._extended())
        return result

    def _extended(self):
        """
        The extension data of a multipart body part are in the following order:

          1. body parameter parenthesized list
               A parenthesized list of attribute/value pairs [e.g., ("foo"
               "bar" "baz" "rag") where "bar" is the value of "foo", and
               "rag" is the value of "baz"] as defined in [MIME-IMB].

          2. body disposition
               A parenthesized list, consisting of a disposition type
               string, followed by a parenthesized list of disposition
               attribute/value pairs as defined in [DISPOSITION].

          3. body language
               A string or parenthesized list giving the body language
               value as defined in [LANGUAGE-TAGS].

          4. body location
               A string list giving the body content URI as defined in
               [LOCATION].
        """
        result = []
        headers = self.message.getHeaders(False, 'content-language', 'content-location', 'content-disposition')
        result.append(self._unquotedAttrs())
        result.append(self._disposition(headers.get('content-disposition')))
        result.append(headers.get('content-language', None))
        result.append(headers.get('content-location', None))
        return result