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
class FzDevice2(FzDevice):
    """ Wrapper class for struct fz_device with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == FzDevice2:
            _self = None
        else:
            _self = self
        _mupdf.FzDevice2_swiginit(self, _mupdf.new_FzDevice2(_self))

    def use_virtual_close_device(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.FzDevice2_use_virtual_close_device(self, use)

    def use_virtual_drop_device(self, use=True):
        return _mupdf.FzDevice2_use_virtual_drop_device(self, use)

    def use_virtual_fill_path(self, use=True):
        return _mupdf.FzDevice2_use_virtual_fill_path(self, use)

    def use_virtual_stroke_path(self, use=True):
        return _mupdf.FzDevice2_use_virtual_stroke_path(self, use)

    def use_virtual_clip_path(self, use=True):
        return _mupdf.FzDevice2_use_virtual_clip_path(self, use)

    def use_virtual_clip_stroke_path(self, use=True):
        return _mupdf.FzDevice2_use_virtual_clip_stroke_path(self, use)

    def use_virtual_fill_text(self, use=True):
        return _mupdf.FzDevice2_use_virtual_fill_text(self, use)

    def use_virtual_stroke_text(self, use=True):
        return _mupdf.FzDevice2_use_virtual_stroke_text(self, use)

    def use_virtual_clip_text(self, use=True):
        return _mupdf.FzDevice2_use_virtual_clip_text(self, use)

    def use_virtual_clip_stroke_text(self, use=True):
        return _mupdf.FzDevice2_use_virtual_clip_stroke_text(self, use)

    def use_virtual_ignore_text(self, use=True):
        return _mupdf.FzDevice2_use_virtual_ignore_text(self, use)

    def use_virtual_fill_shade(self, use=True):
        return _mupdf.FzDevice2_use_virtual_fill_shade(self, use)

    def use_virtual_fill_image(self, use=True):
        return _mupdf.FzDevice2_use_virtual_fill_image(self, use)

    def use_virtual_fill_image_mask(self, use=True):
        return _mupdf.FzDevice2_use_virtual_fill_image_mask(self, use)

    def use_virtual_clip_image_mask(self, use=True):
        return _mupdf.FzDevice2_use_virtual_clip_image_mask(self, use)

    def use_virtual_pop_clip(self, use=True):
        return _mupdf.FzDevice2_use_virtual_pop_clip(self, use)

    def use_virtual_begin_mask(self, use=True):
        return _mupdf.FzDevice2_use_virtual_begin_mask(self, use)

    def use_virtual_end_mask(self, use=True):
        return _mupdf.FzDevice2_use_virtual_end_mask(self, use)

    def use_virtual_begin_group(self, use=True):
        return _mupdf.FzDevice2_use_virtual_begin_group(self, use)

    def use_virtual_end_group(self, use=True):
        return _mupdf.FzDevice2_use_virtual_end_group(self, use)

    def use_virtual_begin_tile(self, use=True):
        return _mupdf.FzDevice2_use_virtual_begin_tile(self, use)

    def use_virtual_end_tile(self, use=True):
        return _mupdf.FzDevice2_use_virtual_end_tile(self, use)

    def use_virtual_render_flags(self, use=True):
        return _mupdf.FzDevice2_use_virtual_render_flags(self, use)

    def use_virtual_set_default_colorspaces(self, use=True):
        return _mupdf.FzDevice2_use_virtual_set_default_colorspaces(self, use)

    def use_virtual_begin_layer(self, use=True):
        return _mupdf.FzDevice2_use_virtual_begin_layer(self, use)

    def use_virtual_end_layer(self, use=True):
        return _mupdf.FzDevice2_use_virtual_end_layer(self, use)

    def use_virtual_begin_structure(self, use=True):
        return _mupdf.FzDevice2_use_virtual_begin_structure(self, use)

    def use_virtual_end_structure(self, use=True):
        return _mupdf.FzDevice2_use_virtual_end_structure(self, use)

    def use_virtual_begin_metatext(self, use=True):
        return _mupdf.FzDevice2_use_virtual_begin_metatext(self, use)

    def use_virtual_end_metatext(self, use=True):
        return _mupdf.FzDevice2_use_virtual_end_metatext(self, use)

    def close_device(self, arg_0):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.FzDevice2_close_device(self, arg_0)

    def drop_device(self, arg_0):
        return _mupdf.FzDevice2_drop_device(self, arg_0)

    def fill_path(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8):
        return _mupdf.FzDevice2_fill_path(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8)

    def stroke_path(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8):
        return _mupdf.FzDevice2_stroke_path(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8)

    def clip_path(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzDevice2_clip_path(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def clip_stroke_path(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzDevice2_clip_stroke_path(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def fill_text(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.FzDevice2_fill_text(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def stroke_text(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8):
        return _mupdf.FzDevice2_stroke_text(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8)

    def clip_text(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.FzDevice2_clip_text(self, arg_0, arg_2, arg_3, arg_4)

    def clip_stroke_text(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzDevice2_clip_stroke_text(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def ignore_text(self, arg_0, arg_2, arg_3):
        return _mupdf.FzDevice2_ignore_text(self, arg_0, arg_2, arg_3)

    def fill_shade(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzDevice2_fill_shade(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def fill_image(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzDevice2_fill_image(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def fill_image_mask(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.FzDevice2_fill_image_mask(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def clip_image_mask(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.FzDevice2_clip_image_mask(self, arg_0, arg_2, arg_3, arg_4)

    def pop_clip(self, arg_0):
        return _mupdf.FzDevice2_pop_clip(self, arg_0)

    def begin_mask(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6):
        return _mupdf.FzDevice2_begin_mask(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6)

    def end_mask(self, arg_0, arg_2):
        return _mupdf.FzDevice2_end_mask(self, arg_0, arg_2)

    def begin_group(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.FzDevice2_begin_group(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def end_group(self, arg_0):
        return _mupdf.FzDevice2_end_group(self, arg_0)

    def begin_tile(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.FzDevice2_begin_tile(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def end_tile(self, arg_0):
        return _mupdf.FzDevice2_end_tile(self, arg_0)

    def render_flags(self, arg_0, arg_2, arg_3):
        return _mupdf.FzDevice2_render_flags(self, arg_0, arg_2, arg_3)

    def set_default_colorspaces(self, arg_0, arg_2):
        return _mupdf.FzDevice2_set_default_colorspaces(self, arg_0, arg_2)

    def begin_layer(self, arg_0, arg_2):
        return _mupdf.FzDevice2_begin_layer(self, arg_0, arg_2)

    def end_layer(self, arg_0):
        return _mupdf.FzDevice2_end_layer(self, arg_0)

    def begin_structure(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.FzDevice2_begin_structure(self, arg_0, arg_2, arg_3, arg_4)

    def end_structure(self, arg_0):
        return _mupdf.FzDevice2_end_structure(self, arg_0)

    def begin_metatext(self, arg_0, arg_2, arg_3):
        return _mupdf.FzDevice2_begin_metatext(self, arg_0, arg_2, arg_3)

    def end_metatext(self, arg_0):
        return _mupdf.FzDevice2_end_metatext(self, arg_0)
    __swig_destroy__ = _mupdf.delete_FzDevice2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_FzDevice2(self)
        return weakref.proxy(self)