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