import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _make_failed_test(methodname, exception, suiteClass, message):
    test = _FailedTest(methodname, exception)
    return (suiteClass((test,)), message)