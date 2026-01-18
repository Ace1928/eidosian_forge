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
class fz_shaper_data_t(object):
    """
    In order to shape a given font, we need to
    declare it to a shaper library (harfbuzz, by default, but others
    are possible). To avoid redeclaring it every time we need to
    shape, we hold a shaper handle and the destructor for it within
    the font itself. The handle is initialised by the caller when
    first required and the destructor is called when the fz_font is
    destroyed.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    shaper_handle = property(_mupdf.fz_shaper_data_t_shaper_handle_get, _mupdf.fz_shaper_data_t_shaper_handle_set)
    destroy = property(_mupdf.fz_shaper_data_t_destroy_get, _mupdf.fz_shaper_data_t_destroy_set)

    def __init__(self):
        _mupdf.fz_shaper_data_t_swiginit(self, _mupdf.new_fz_shaper_data_t())
    __swig_destroy__ = _mupdf.delete_fz_shaper_data_t