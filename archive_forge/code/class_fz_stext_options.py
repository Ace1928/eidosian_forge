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
class fz_stext_options(object):
    """	Options for creating structured text."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    flags = property(_mupdf.fz_stext_options_flags_get, _mupdf.fz_stext_options_flags_set)
    scale = property(_mupdf.fz_stext_options_scale_get, _mupdf.fz_stext_options_scale_set)

    def __init__(self):
        _mupdf.fz_stext_options_swiginit(self, _mupdf.new_fz_stext_options())
    __swig_destroy__ = _mupdf.delete_fz_stext_options