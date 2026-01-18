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
@classmethod
def _get_skip_backend_reason(cls, backend):
    """
        make sure os_crypt backend is tested
        when it's known os_crypt will be faked by _patch_safe_crypt()
        """
    assert backend == 'os_crypt'
    reason = super(OsCryptMixin, cls)._get_skip_backend_reason(backend)
    from passlib.utils import has_crypt
    if reason == cls._BACKEND_NOT_AVAILABLE and has_crypt:
        if TEST_MODE('full') and cls._get_safe_crypt_handler_backend()[1]:
            return None
        else:
            return 'hash not supported by os crypt()'
    return reason