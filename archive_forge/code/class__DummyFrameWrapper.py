from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
class _DummyFrameWrapper(object):

    def __init__(self, frame, f_lineno, f_back):
        self._base_frame = frame
        self.f_lineno = f_lineno
        self.f_back = f_back
        self.f_trace = None
        original_code = frame.f_code
        name = original_code.co_name
        self.f_code = FCode(name, original_code.co_filename)

    @property
    def f_locals(self):
        return self._base_frame.f_locals

    @property
    def f_globals(self):
        return self._base_frame.f_globals

    def __str__(self):
        return "<_DummyFrameWrapper, file '%s', line %s, %s" % (self.f_code.co_filename, self.f_lineno, self.f_code.co_name)
    __repr__ = __str__