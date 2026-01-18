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
class _AssertWarningList(warnings.catch_warnings):
    """context manager for assertWarningList()"""

    def __init__(self, case, **kwds):
        self.case = case
        self.kwds = kwds
        self.__super = super(TestCase._AssertWarningList, self)
        self.__super.__init__(record=True)

    def __enter__(self):
        self.log = self.__super.__enter__()

    def __exit__(self, *exc_info):
        self.__super.__exit__(*exc_info)
        if exc_info[0] is None:
            self.case.assertWarningList(self.log, **self.kwds)