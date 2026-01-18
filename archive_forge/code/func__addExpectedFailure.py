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
def _addExpectedFailure(self, result, exc_info):
    try:
        addExpectedFailure = result.addExpectedFailure
    except AttributeError:
        warnings.warn('TestResult has no addExpectedFailure method, reporting as passes', RuntimeWarning)
        result.addSuccess(self)
    else:
        addExpectedFailure(self, exc_info)