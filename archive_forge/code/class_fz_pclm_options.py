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
class fz_pclm_options(object):
    """	PCLm output"""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    compress = property(_mupdf.fz_pclm_options_compress_get, _mupdf.fz_pclm_options_compress_set)
    strip_height = property(_mupdf.fz_pclm_options_strip_height_get, _mupdf.fz_pclm_options_strip_height_set)
    page_count = property(_mupdf.fz_pclm_options_page_count_get, _mupdf.fz_pclm_options_page_count_set)

    def __init__(self):
        _mupdf.fz_pclm_options_swiginit(self, _mupdf.new_fz_pclm_options())
    __swig_destroy__ = _mupdf.delete_fz_pclm_options