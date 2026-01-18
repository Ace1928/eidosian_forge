import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _make_failed_load_tests(name, exception, suiteClass):
    message = 'Failed to call load_tests:\n%s' % (traceback.format_exc(),)
    return _make_failed_test(name, exception, suiteClass, message)