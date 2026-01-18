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
class fz_device(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_device_refs_get, _mupdf.fz_device_refs_set)
    hints = property(_mupdf.fz_device_hints_get, _mupdf.fz_device_hints_set)
    flags = property(_mupdf.fz_device_flags_get, _mupdf.fz_device_flags_set)
    close_device = property(_mupdf.fz_device_close_device_get, _mupdf.fz_device_close_device_set)
    drop_device = property(_mupdf.fz_device_drop_device_get, _mupdf.fz_device_drop_device_set)
    fill_path = property(_mupdf.fz_device_fill_path_get, _mupdf.fz_device_fill_path_set)
    stroke_path = property(_mupdf.fz_device_stroke_path_get, _mupdf.fz_device_stroke_path_set)
    clip_path = property(_mupdf.fz_device_clip_path_get, _mupdf.fz_device_clip_path_set)
    clip_stroke_path = property(_mupdf.fz_device_clip_stroke_path_get, _mupdf.fz_device_clip_stroke_path_set)
    fill_text = property(_mupdf.fz_device_fill_text_get, _mupdf.fz_device_fill_text_set)
    stroke_text = property(_mupdf.fz_device_stroke_text_get, _mupdf.fz_device_stroke_text_set)
    clip_text = property(_mupdf.fz_device_clip_text_get, _mupdf.fz_device_clip_text_set)
    clip_stroke_text = property(_mupdf.fz_device_clip_stroke_text_get, _mupdf.fz_device_clip_stroke_text_set)
    ignore_text = property(_mupdf.fz_device_ignore_text_get, _mupdf.fz_device_ignore_text_set)
    fill_shade = property(_mupdf.fz_device_fill_shade_get, _mupdf.fz_device_fill_shade_set)
    fill_image = property(_mupdf.fz_device_fill_image_get, _mupdf.fz_device_fill_image_set)
    fill_image_mask = property(_mupdf.fz_device_fill_image_mask_get, _mupdf.fz_device_fill_image_mask_set)
    clip_image_mask = property(_mupdf.fz_device_clip_image_mask_get, _mupdf.fz_device_clip_image_mask_set)
    pop_clip = property(_mupdf.fz_device_pop_clip_get, _mupdf.fz_device_pop_clip_set)
    begin_mask = property(_mupdf.fz_device_begin_mask_get, _mupdf.fz_device_begin_mask_set)
    end_mask = property(_mupdf.fz_device_end_mask_get, _mupdf.fz_device_end_mask_set)
    begin_group = property(_mupdf.fz_device_begin_group_get, _mupdf.fz_device_begin_group_set)
    end_group = property(_mupdf.fz_device_end_group_get, _mupdf.fz_device_end_group_set)
    begin_tile = property(_mupdf.fz_device_begin_tile_get, _mupdf.fz_device_begin_tile_set)
    end_tile = property(_mupdf.fz_device_end_tile_get, _mupdf.fz_device_end_tile_set)
    render_flags = property(_mupdf.fz_device_render_flags_get, _mupdf.fz_device_render_flags_set)
    set_default_colorspaces = property(_mupdf.fz_device_set_default_colorspaces_get, _mupdf.fz_device_set_default_colorspaces_set)
    begin_layer = property(_mupdf.fz_device_begin_layer_get, _mupdf.fz_device_begin_layer_set)
    end_layer = property(_mupdf.fz_device_end_layer_get, _mupdf.fz_device_end_layer_set)
    begin_structure = property(_mupdf.fz_device_begin_structure_get, _mupdf.fz_device_begin_structure_set)
    end_structure = property(_mupdf.fz_device_end_structure_get, _mupdf.fz_device_end_structure_set)
    begin_metatext = property(_mupdf.fz_device_begin_metatext_get, _mupdf.fz_device_begin_metatext_set)
    end_metatext = property(_mupdf.fz_device_end_metatext_get, _mupdf.fz_device_end_metatext_set)
    d1_rect = property(_mupdf.fz_device_d1_rect_get, _mupdf.fz_device_d1_rect_set)
    container_len = property(_mupdf.fz_device_container_len_get, _mupdf.fz_device_container_len_set)
    container_cap = property(_mupdf.fz_device_container_cap_get, _mupdf.fz_device_container_cap_set)
    container = property(_mupdf.fz_device_container_get, _mupdf.fz_device_container_set)

    def __init__(self):
        _mupdf.fz_device_swiginit(self, _mupdf.new_fz_device())
    __swig_destroy__ = _mupdf.delete_fz_device