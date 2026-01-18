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
class fz_range(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    offset = property(_mupdf.fz_range_offset_get, _mupdf.fz_range_offset_set)
    length = property(_mupdf.fz_range_length_get, _mupdf.fz_range_length_set)

    def __init__(self):
        _mupdf.fz_range_swiginit(self, _mupdf.new_fz_range())
    __swig_destroy__ = _mupdf.delete_fz_range