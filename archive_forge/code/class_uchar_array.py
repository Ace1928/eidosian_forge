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
class uchar_array(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, nelements):
        _mupdf.uchar_array_swiginit(self, _mupdf.new_uchar_array(nelements))
    __swig_destroy__ = _mupdf.delete_uchar_array

    def __getitem__(self, index):
        return _mupdf.uchar_array___getitem__(self, index)

    def __setitem__(self, index, value):
        return _mupdf.uchar_array___setitem__(self, index, value)

    def cast(self):
        return _mupdf.uchar_array_cast(self)

    @staticmethod
    def frompointer(t):
        return _mupdf.uchar_array_frompointer(t)