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
class fz_aa_context(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    hscale = property(_mupdf.fz_aa_context_hscale_get, _mupdf.fz_aa_context_hscale_set)
    vscale = property(_mupdf.fz_aa_context_vscale_get, _mupdf.fz_aa_context_vscale_set)
    scale = property(_mupdf.fz_aa_context_scale_get, _mupdf.fz_aa_context_scale_set)
    bits = property(_mupdf.fz_aa_context_bits_get, _mupdf.fz_aa_context_bits_set)
    text_bits = property(_mupdf.fz_aa_context_text_bits_get, _mupdf.fz_aa_context_text_bits_set)
    min_line_width = property(_mupdf.fz_aa_context_min_line_width_get, _mupdf.fz_aa_context_min_line_width_set)

    def __init__(self):
        _mupdf.fz_aa_context_swiginit(self, _mupdf.new_fz_aa_context())
    __swig_destroy__ = _mupdf.delete_fz_aa_context