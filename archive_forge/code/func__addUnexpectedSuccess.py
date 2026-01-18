import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
def _addUnexpectedSuccess(self, result):
    try:
        addUnexpectedSuccess = result.addUnexpectedSuccess
    except AttributeError:
        warnings.warn('TestResult has no addUnexpectedSuccess method, reporting as failure', RuntimeWarning)
        try:
            raise _UnexpectedSuccess from None
        except _UnexpectedSuccess:
            result.addFailure(self, sys.exc_info())
    else:
        addUnexpectedSuccess(self)