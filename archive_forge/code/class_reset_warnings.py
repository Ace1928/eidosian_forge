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
class reset_warnings(warnings.catch_warnings):
    """catch_warnings() wrapper which clears warning registry & filters"""

    def __init__(self, reset_filter='always', reset_registry='.*', **kwds):
        super(reset_warnings, self).__init__(**kwds)
        self._reset_filter = reset_filter
        self._reset_registry = re.compile(reset_registry) if reset_registry else None

    def __enter__(self):
        ret = super(reset_warnings, self).__enter__()
        if self._reset_filter:
            warnings.resetwarnings()
            warnings.simplefilter(self._reset_filter)
        pattern = self._reset_registry
        if pattern:
            backup = self._orig_registry = {}
            for name, mod in list(sys.modules.items()):
                if mod is None or not pattern.match(name):
                    continue
                reg = getattr(mod, '__warningregistry__', None)
                if reg:
                    backup[name] = reg.copy()
                    reg.clear()
        return ret

    def __exit__(self, *exc_info):
        pattern = self._reset_registry
        if pattern:
            backup = self._orig_registry
            for name, mod in list(sys.modules.items()):
                if mod is None or not pattern.match(name):
                    continue
                reg = getattr(mod, '__warningregistry__', None)
                if reg:
                    reg.clear()
                orig = backup.get(name)
                if orig:
                    if reg is None:
                        setattr(mod, '__warningregistry__', orig)
                    else:
                        reg.update(orig)
        super(reset_warnings, self).__exit__(*exc_info)