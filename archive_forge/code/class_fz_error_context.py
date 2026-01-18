from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_error_context(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    top = property(_mupdf.fz_error_context_top_get, _mupdf.fz_error_context_top_set)
    stack = property(_mupdf.fz_error_context_stack_get, _mupdf.fz_error_context_stack_set)
    padding = property(_mupdf.fz_error_context_padding_get, _mupdf.fz_error_context_padding_set)
    stack_base = property(_mupdf.fz_error_context_stack_base_get, _mupdf.fz_error_context_stack_base_set)
    errcode = property(_mupdf.fz_error_context_errcode_get, _mupdf.fz_error_context_errcode_set)
    errnum = property(_mupdf.fz_error_context_errnum_get, _mupdf.fz_error_context_errnum_set)
    print_user = property(_mupdf.fz_error_context_print_user_get, _mupdf.fz_error_context_print_user_set)
    _print = property(_mupdf.fz_error_context__print_get, _mupdf.fz_error_context__print_set)
    message = property(_mupdf.fz_error_context_message_get, _mupdf.fz_error_context_message_set)

    def __init__(self):
        _mupdf.fz_error_context_swiginit(self, _mupdf.new_fz_error_context())
    __swig_destroy__ = _mupdf.delete_fz_error_context