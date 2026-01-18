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
def _getContentType(msg):
    """
    Return a two-tuple of the main and subtype of the given message.
    """
    attrs = None
    mm = msg.getHeaders(False, 'content-type').get('content-type', '')
    mm = ''.join(mm.splitlines())
    if mm:
        mimetype = mm.split(';')
        type = mimetype[0].split('/', 1)
        if len(type) == 1:
            major = type[0]
            minor = None
        else:
            major, minor = type
        attrs = dict((x.strip().lower().split('=', 1) for x in mimetype[1:]))
    else:
        major = minor = None
    return (major, minor, attrs)