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
def _patch_safe_crypt(self):
    """if crypt() doesn't support current hash alg, this patches
        safe_crypt() so that it transparently uses another one of the handler's
        backends, so that we can go ahead and test as much of code path
        as possible.
        """
    handler, alt_backend = self._get_safe_crypt_handler_backend()
    if not alt_backend:
        raise AssertionError('handler has no available alternate backends!')
    alt_handler = handler.using()
    alt_handler.set_backend(alt_backend)

    def crypt_stub(secret, hash):
        hash = alt_handler.genhash(secret, hash)
        assert isinstance(hash, str)
        return hash
    import passlib.utils as mod
    self.patchAttr(mod, '_crypt', crypt_stub)
    self.using_patched_crypt = True