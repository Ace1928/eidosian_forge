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
class fz_stext_char(object):
    """
    A text char is a unicode character, the style in which is
    appears, and the point at which it is positioned.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    c = property(_mupdf.fz_stext_char_c_get, _mupdf.fz_stext_char_c_set)
    bidi = property(_mupdf.fz_stext_char_bidi_get, _mupdf.fz_stext_char_bidi_set)
    color = property(_mupdf.fz_stext_char_color_get, _mupdf.fz_stext_char_color_set)
    origin = property(_mupdf.fz_stext_char_origin_get, _mupdf.fz_stext_char_origin_set)
    quad = property(_mupdf.fz_stext_char_quad_get, _mupdf.fz_stext_char_quad_set)
    size = property(_mupdf.fz_stext_char_size_get, _mupdf.fz_stext_char_size_set)
    font = property(_mupdf.fz_stext_char_font_get, _mupdf.fz_stext_char_font_set)
    next = property(_mupdf.fz_stext_char_next_get, _mupdf.fz_stext_char_next_set)

    def __init__(self):
        _mupdf.fz_stext_char_swiginit(self, _mupdf.new_fz_stext_char())
    __swig_destroy__ = _mupdf.delete_fz_stext_char