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
class fz_draw_options(object):
    """
    struct fz_draw_options: Options for creating a pixmap and draw
    device.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    rotate = property(_mupdf.fz_draw_options_rotate_get, _mupdf.fz_draw_options_rotate_set)
    x_resolution = property(_mupdf.fz_draw_options_x_resolution_get, _mupdf.fz_draw_options_x_resolution_set)
    y_resolution = property(_mupdf.fz_draw_options_y_resolution_get, _mupdf.fz_draw_options_y_resolution_set)
    width = property(_mupdf.fz_draw_options_width_get, _mupdf.fz_draw_options_width_set)
    height = property(_mupdf.fz_draw_options_height_get, _mupdf.fz_draw_options_height_set)
    colorspace = property(_mupdf.fz_draw_options_colorspace_get, _mupdf.fz_draw_options_colorspace_set)
    alpha = property(_mupdf.fz_draw_options_alpha_get, _mupdf.fz_draw_options_alpha_set)
    graphics = property(_mupdf.fz_draw_options_graphics_get, _mupdf.fz_draw_options_graphics_set)
    text = property(_mupdf.fz_draw_options_text_get, _mupdf.fz_draw_options_text_set)

    def __init__(self):
        _mupdf.fz_draw_options_swiginit(self, _mupdf.new_fz_draw_options())
    __swig_destroy__ = _mupdf.delete_fz_draw_options