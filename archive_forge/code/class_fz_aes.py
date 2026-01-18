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
class fz_aes(object):
    """
    Structure definitions are public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    nr = property(_mupdf.fz_aes_nr_get, _mupdf.fz_aes_nr_set)
    rk = property(_mupdf.fz_aes_rk_get, _mupdf.fz_aes_rk_set)
    buf = property(_mupdf.fz_aes_buf_get, _mupdf.fz_aes_buf_set)

    def __init__(self):
        _mupdf.fz_aes_swiginit(self, _mupdf.new_fz_aes())
    __swig_destroy__ = _mupdf.delete_fz_aes