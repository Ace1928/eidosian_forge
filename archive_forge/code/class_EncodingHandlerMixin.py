from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
class EncodingHandlerMixin(HandlerCase):
    """helper for handlers w/ 'encoding' context kwd; mixin for HandlerCase

    this overrides the HandlerCase test harness methods
    so that an encoding can be inserted to hash/verify
    calls by passing in a pair of strings as the password
    will be interpreted as (secret,encoding)
    """
    __unittest_skip = True
    stock_passwords = [u('test'), b'test', u('¬º')]

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):
        password_alphabet = u('qwerty1234<>.@*#! ¬')

    def populate_context(self, secret, kwds):
        """insert encoding into kwds"""
        if isinstance(secret, tuple):
            secret, encoding = secret
            kwds.setdefault('encoding', encoding)
        return secret