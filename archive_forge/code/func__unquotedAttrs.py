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
def _unquotedAttrs(self):
    """
        @return: The I{Content-Type} parameters, unquoted, as a flat list with
            each Nth element giving a parameter name and N+1th element giving
            the corresponding parameter value.
        """
    if self.attrs:
        unquoted = [(k, unquote(v)) for k, v in self.attrs.items()]
        return [y for x in sorted(unquoted) for y in x]
    return None