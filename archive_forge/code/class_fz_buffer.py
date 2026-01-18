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
class fz_buffer(object):
    """
    fz_buffer is a wrapper around a dynamically allocated array of
    bytes.

    Buffers have a capacity (the number of bytes storage immediately
    available) and a current size.

    The contents of the structure are considered implementation
    details and are subject to change. Users should use the accessor
    functions in preference.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_buffer_refs_get, _mupdf.fz_buffer_refs_set)
    data = property(_mupdf.fz_buffer_data_get, _mupdf.fz_buffer_data_set)
    cap = property(_mupdf.fz_buffer_cap_get, _mupdf.fz_buffer_cap_set)
    len = property(_mupdf.fz_buffer_len_get, _mupdf.fz_buffer_len_set)
    unused_bits = property(_mupdf.fz_buffer_unused_bits_get, _mupdf.fz_buffer_unused_bits_set)
    shared = property(_mupdf.fz_buffer_shared_get, _mupdf.fz_buffer_shared_set)

    def __init__(self):
        _mupdf.fz_buffer_swiginit(self, _mupdf.new_fz_buffer())
    __swig_destroy__ = _mupdf.delete_fz_buffer