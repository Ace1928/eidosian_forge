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
class fz_bitmap(object):
    """
    Bitmaps have 1 bit per component. Only used for creating
    halftoned versions of contone buffers, and saving out. Samples
    are stored msb first, akin to pbms.

    The internals of this struct are considered implementation
    details and subject to change. Where possible, accessor
    functions should be used in preference.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_bitmap_refs_get, _mupdf.fz_bitmap_refs_set)
    w = property(_mupdf.fz_bitmap_w_get, _mupdf.fz_bitmap_w_set)
    h = property(_mupdf.fz_bitmap_h_get, _mupdf.fz_bitmap_h_set)
    stride = property(_mupdf.fz_bitmap_stride_get, _mupdf.fz_bitmap_stride_set)
    n = property(_mupdf.fz_bitmap_n_get, _mupdf.fz_bitmap_n_set)
    xres = property(_mupdf.fz_bitmap_xres_get, _mupdf.fz_bitmap_xres_set)
    yres = property(_mupdf.fz_bitmap_yres_get, _mupdf.fz_bitmap_yres_set)
    samples = property(_mupdf.fz_bitmap_samples_get, _mupdf.fz_bitmap_samples_set)

    def __init__(self):
        _mupdf.fz_bitmap_swiginit(self, _mupdf.new_fz_bitmap())
    __swig_destroy__ = _mupdf.delete_fz_bitmap