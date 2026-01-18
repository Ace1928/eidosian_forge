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
class fz_colorspace(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    key_storable = property(_mupdf.fz_colorspace_key_storable_get, _mupdf.fz_colorspace_key_storable_set)
    type = property(_mupdf.fz_colorspace_type_get, _mupdf.fz_colorspace_type_set)
    flags = property(_mupdf.fz_colorspace_flags_get, _mupdf.fz_colorspace_flags_set)
    n = property(_mupdf.fz_colorspace_n_get, _mupdf.fz_colorspace_n_set)
    name = property(_mupdf.fz_colorspace_name_get, _mupdf.fz_colorspace_name_set)

    def __init__(self):
        _mupdf.fz_colorspace_swiginit(self, _mupdf.new_fz_colorspace())
    __swig_destroy__ = _mupdf.delete_fz_colorspace