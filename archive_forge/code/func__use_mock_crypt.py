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
def _use_mock_crypt(self):
    """
        patch passlib.utils.safe_crypt() so it returns mock value for duration of test.
        returns function whose .return_value controls what's returned.
        this defaults to None.
        """
    import passlib.utils as mod

    def mock_crypt(secret, config):
        if secret == 'test':
            return mock_crypt.__wrapped__(secret, config)
        else:
            return mock_crypt.return_value
    mock_crypt.__wrapped__ = mod._crypt
    mock_crypt.return_value = None
    self.patchAttr(mod, '_crypt', mock_crypt)
    return mock_crypt