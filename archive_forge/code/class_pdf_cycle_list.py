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
class pdf_cycle_list(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    up = property(_mupdf.pdf_cycle_list_up_get, _mupdf.pdf_cycle_list_up_set)
    num = property(_mupdf.pdf_cycle_list_num_get, _mupdf.pdf_cycle_list_num_set)

    def __init__(self):
        _mupdf.pdf_cycle_list_swiginit(self, _mupdf.new_pdf_cycle_list())
    __swig_destroy__ = _mupdf.delete_pdf_cycle_list