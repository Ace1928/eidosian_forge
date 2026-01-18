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
class fz_irect(object):
    """
    fz_irect is a rectangle using integers instead of floats.

    It's used in the draw device and for pixmap dimensions.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x0 = property(_mupdf.fz_irect_x0_get, _mupdf.fz_irect_x0_set)
    y0 = property(_mupdf.fz_irect_y0_get, _mupdf.fz_irect_y0_set)
    x1 = property(_mupdf.fz_irect_x1_get, _mupdf.fz_irect_x1_set)
    y1 = property(_mupdf.fz_irect_y1_get, _mupdf.fz_irect_y1_set)

    def __init__(self):
        _mupdf.fz_irect_swiginit(self, _mupdf.new_fz_irect())
    __swig_destroy__ = _mupdf.delete_fz_irect