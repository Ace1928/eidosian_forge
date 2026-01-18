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
def _fetch(self, messages, useUID=0, **terms):
    messages = str(messages).encode('ascii')
    fetch = useUID and b'UID FETCH' or b'FETCH'
    if 'rfc822text' in terms:
        del terms['rfc822text']
        terms['rfc822.text'] = True
    if 'rfc822size' in terms:
        del terms['rfc822size']
        terms['rfc822.size'] = True
    if 'rfc822header' in terms:
        del terms['rfc822header']
        terms['rfc822.header'] = True
    encodedTerms = [networkString(s) for s in terms]
    cmd = messages + b' (' + b' '.join([s.upper() for s in encodedTerms]) + b')'
    d = self.sendCommand(Command(fetch, cmd, wantResponse=(b'FETCH',)))
    d.addCallback(self._cbFetch, [t.upper() for t in terms.keys()], True)
    return d