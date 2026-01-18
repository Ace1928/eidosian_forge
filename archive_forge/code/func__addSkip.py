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
def _addSkip(result, test_case, reason):
    addSkip = getattr(result, 'addSkip', None)
    if addSkip is not None:
        addSkip(test_case, reason)
    else:
        warnings.warn('TestResult has no addSkip method, skips not reported', RuntimeWarning, 2)
        result.addSuccess(test_case)