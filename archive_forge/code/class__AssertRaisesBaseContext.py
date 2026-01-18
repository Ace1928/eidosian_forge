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
class _AssertRaisesBaseContext(_BaseTestCaseContext):

    def __init__(self, expected, test_case, expected_regex=None):
        _BaseTestCaseContext.__init__(self, test_case)
        self.expected = expected
        self.test_case = test_case
        if expected_regex is not None:
            expected_regex = re.compile(expected_regex)
        self.expected_regex = expected_regex
        self.obj_name = None
        self.msg = None

    def handle(self, name, args, kwargs):
        """
        If args is empty, assertRaises/Warns is being used as a
        context manager, so check for a 'msg' kwarg and return self.
        If args is not empty, call a callable passing positional and keyword
        arguments.
        """
        try:
            if not _is_subtype(self.expected, self._base_type):
                raise TypeError('%s() arg 1 must be %s' % (name, self._base_type_str))
            if not args:
                self.msg = kwargs.pop('msg', None)
                if kwargs:
                    raise TypeError('%r is an invalid keyword argument for this function' % (next(iter(kwargs)),))
                return self
            callable_obj, *args = args
            try:
                self.obj_name = callable_obj.__name__
            except AttributeError:
                self.obj_name = str(callable_obj)
            with self:
                callable_obj(*args, **kwargs)
        finally:
            self = None