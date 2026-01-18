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
class fz_function(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.fz_function_storable_get, _mupdf.fz_function_storable_set)
    size = property(_mupdf.fz_function_size_get, _mupdf.fz_function_size_set)
    m = property(_mupdf.fz_function_m_get, _mupdf.fz_function_m_set)
    n = property(_mupdf.fz_function_n_get, _mupdf.fz_function_n_set)
    eval = property(_mupdf.fz_function_eval_get, _mupdf.fz_function_eval_set)

    def __init__(self):
        _mupdf.fz_function_swiginit(self, _mupdf.new_fz_function())
    __swig_destroy__ = _mupdf.delete_fz_function