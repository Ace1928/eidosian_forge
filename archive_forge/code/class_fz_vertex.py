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
class fz_vertex(object):
    """	Handy routine for processing mesh based shades"""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    p = property(_mupdf.fz_vertex_p_get, _mupdf.fz_vertex_p_set)
    c = property(_mupdf.fz_vertex_c_get, _mupdf.fz_vertex_c_set)

    def __init__(self):
        _mupdf.fz_vertex_swiginit(self, _mupdf.new_fz_vertex())
    __swig_destroy__ = _mupdf.delete_fz_vertex