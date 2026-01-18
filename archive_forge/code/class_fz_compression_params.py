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
class fz_compression_params(object):
    """
    Compression parameters used for buffers of compressed data;
    typically for the source data for images.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    type = property(_mupdf.fz_compression_params_type_get, _mupdf.fz_compression_params_type_set)

    def __init__(self):
        _mupdf.fz_compression_params_swiginit(self, _mupdf.new_fz_compression_params())
    __swig_destroy__ = _mupdf.delete_fz_compression_params