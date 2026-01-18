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
class _AssertWarnsContext(_AssertRaisesBaseContext):
    """A context manager used to implement TestCase.assertWarns* methods."""
    _base_type = Warning
    _base_type_str = 'a warning type or tuple of warning types'

    def __enter__(self):
        for v in list(sys.modules.values()):
            if getattr(v, '__warningregistry__', None):
                v.__warningregistry__ = {}
        self.warnings_manager = warnings.catch_warnings(record=True)
        self.warnings = self.warnings_manager.__enter__()
        warnings.simplefilter('always', self.expected)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.warnings_manager.__exit__(exc_type, exc_value, tb)
        if exc_type is not None:
            return
        try:
            exc_name = self.expected.__name__
        except AttributeError:
            exc_name = str(self.expected)
        first_matching = None
        for m in self.warnings:
            w = m.message
            if not isinstance(w, self.expected):
                continue
            if first_matching is None:
                first_matching = w
            if self.expected_regex is not None and (not self.expected_regex.search(str(w))):
                continue
            self.warning = w
            self.filename = m.filename
            self.lineno = m.lineno
            return
        if first_matching is not None:
            self._raiseFailure('"{}" does not match "{}"'.format(self.expected_regex.pattern, str(first_matching)))
        if self.obj_name:
            self._raiseFailure('{} not triggered by {}'.format(exc_name, self.obj_name))
        else:
            self._raiseFailure('{} not triggered'.format(exc_name))