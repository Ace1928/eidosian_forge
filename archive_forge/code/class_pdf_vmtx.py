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
class pdf_vmtx(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    lo = property(_mupdf.pdf_vmtx_lo_get, _mupdf.pdf_vmtx_lo_set)
    hi = property(_mupdf.pdf_vmtx_hi_get, _mupdf.pdf_vmtx_hi_set)
    x = property(_mupdf.pdf_vmtx_x_get, _mupdf.pdf_vmtx_x_set)
    y = property(_mupdf.pdf_vmtx_y_get, _mupdf.pdf_vmtx_y_set)
    w = property(_mupdf.pdf_vmtx_w_get, _mupdf.pdf_vmtx_w_set)

    def __init__(self):
        _mupdf.pdf_vmtx_swiginit(self, _mupdf.new_pdf_vmtx())
    __swig_destroy__ = _mupdf.delete_pdf_vmtx