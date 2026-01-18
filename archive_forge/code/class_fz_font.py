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
class fz_font(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_font_refs_get, _mupdf.fz_font_refs_set)
    name = property(_mupdf.fz_font_name_get, _mupdf.fz_font_name_set)
    buffer = property(_mupdf.fz_font_buffer_get, _mupdf.fz_font_buffer_set)
    flags = property(_mupdf.fz_font_flags_get, _mupdf.fz_font_flags_set)
    ft_face = property(_mupdf.fz_font_ft_face_get, _mupdf.fz_font_ft_face_set)
    shaper_data = property(_mupdf.fz_font_shaper_data_get, _mupdf.fz_font_shaper_data_set)
    t3matrix = property(_mupdf.fz_font_t3matrix_get, _mupdf.fz_font_t3matrix_set)
    t3resources = property(_mupdf.fz_font_t3resources_get, _mupdf.fz_font_t3resources_set)
    t3procs = property(_mupdf.fz_font_t3procs_get, _mupdf.fz_font_t3procs_set)
    t3lists = property(_mupdf.fz_font_t3lists_get, _mupdf.fz_font_t3lists_set)
    t3widths = property(_mupdf.fz_font_t3widths_get, _mupdf.fz_font_t3widths_set)
    t3flags = property(_mupdf.fz_font_t3flags_get, _mupdf.fz_font_t3flags_set)
    t3doc = property(_mupdf.fz_font_t3doc_get, _mupdf.fz_font_t3doc_set)
    t3run = property(_mupdf.fz_font_t3run_get, _mupdf.fz_font_t3run_set)
    t3freeres = property(_mupdf.fz_font_t3freeres_get, _mupdf.fz_font_t3freeres_set)
    bbox = property(_mupdf.fz_font_bbox_get, _mupdf.fz_font_bbox_set)
    glyph_count = property(_mupdf.fz_font_glyph_count_get, _mupdf.fz_font_glyph_count_set)
    bbox_table = property(_mupdf.fz_font_bbox_table_get, _mupdf.fz_font_bbox_table_set)
    use_glyph_bbox = property(_mupdf.fz_font_use_glyph_bbox_get, _mupdf.fz_font_use_glyph_bbox_set)
    width_count = property(_mupdf.fz_font_width_count_get, _mupdf.fz_font_width_count_set)
    width_default = property(_mupdf.fz_font_width_default_get, _mupdf.fz_font_width_default_set)
    width_table = property(_mupdf.fz_font_width_table_get, _mupdf.fz_font_width_table_set)
    advance_cache = property(_mupdf.fz_font_advance_cache_get, _mupdf.fz_font_advance_cache_set)
    encoding_cache = property(_mupdf.fz_font_encoding_cache_get, _mupdf.fz_font_encoding_cache_set)
    has_digest = property(_mupdf.fz_font_has_digest_get, _mupdf.fz_font_has_digest_set)
    digest = property(_mupdf.fz_font_digest_get, _mupdf.fz_font_digest_set)
    subfont = property(_mupdf.fz_font_subfont_get, _mupdf.fz_font_subfont_set)

    def __init__(self):
        _mupdf.fz_font_swiginit(self, _mupdf.new_fz_font())
    __swig_destroy__ = _mupdf.delete_fz_font