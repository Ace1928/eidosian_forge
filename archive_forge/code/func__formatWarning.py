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
def _formatWarning(self, entry):
    tail = ''
    if hasattr(entry, 'message'):
        tail = ' filename=%r lineno=%r' % (entry.filename, entry.lineno)
        if entry.line:
            tail += ' line=%r' % (entry.line,)
        entry = entry.message
    cls = type(entry)
    return '<%s.%s message=%r%s>' % (cls.__module__, cls.__name__, str(entry), tail)