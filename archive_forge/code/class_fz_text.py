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
class fz_text(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_text_refs_get, _mupdf.fz_text_refs_set)
    head = property(_mupdf.fz_text_head_get, _mupdf.fz_text_head_set)
    tail = property(_mupdf.fz_text_tail_get, _mupdf.fz_text_tail_set)

    def __init__(self):
        _mupdf.fz_text_swiginit(self, _mupdf.new_fz_text())
    __swig_destroy__ = _mupdf.delete_fz_text