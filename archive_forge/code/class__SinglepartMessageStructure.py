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
class _SinglepartMessageStructure(_MessageStructure):
    """
    L{_SinglepartMessageStructure} represents the message structure of a
    non-I{multipart/*} message.
    """
    _HEADERS = ['content-id', 'content-description', 'content-transfer-encoding']

    def __init__(self, message, main, subtype, attrs):
        """
        @param message: An L{IMessagePart} provider which this structure object
            reports on.

        @param main: A L{str} giving the main MIME type of the message (for
            example, C{"text"}).

        @param subtype: A L{str} giving the MIME subtype of the message (for
            example, C{"plain"}).

        @param attrs: A C{dict} giving the parameters of the I{Content-Type}
            header of the message.
        """
        _MessageStructure.__init__(self, message, attrs)
        self.main = main
        self.subtype = subtype
        self.attrs = attrs

    def _basicFields(self):
        """
        Return a list of the basic fields for a single-part message.
        """
        headers = self.message.getHeaders(False, *self._HEADERS)
        size = self.message.getSize()
        major, minor = (self.main, self.subtype)
        unquotedAttrs = self._unquotedAttrs()
        return [major, minor, unquotedAttrs, headers.get('content-id'), headers.get('content-description'), headers.get('content-transfer-encoding'), size]

    def encode(self, extended):
        """
        Construct and return a list of the basic and extended fields for a
        single-part message.  The list suitable to be encoded into a BODY or
        BODYSTRUCTURE response.
        """
        result = self._basicFields()
        if extended:
            result.extend(self._extended())
        return result

    def _extended(self):
        """
        The extension data of a non-multipart body part are in the
        following order:

          1. body MD5

             A string giving the body MD5 value as defined in [MD5].

          2. body disposition

             A parenthesized list with the same content and function as
             the body disposition for a multipart body part.

          3. body language

             A string or parenthesized list giving the body language
             value as defined in [LANGUAGE-TAGS].

          4. body location

             A string list giving the body content URI as defined in
             [LOCATION].

        """
        result = []
        headers = self.message.getHeaders(False, 'content-md5', 'content-disposition', 'content-language', 'content-language')
        result.append(headers.get('content-md5'))
        result.append(self._disposition(headers.get('content-disposition')))
        result.append(headers.get('content-language'))
        result.append(headers.get('content-location'))
        return result