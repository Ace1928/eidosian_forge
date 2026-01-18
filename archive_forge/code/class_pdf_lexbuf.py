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
class pdf_lexbuf(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    size = property(_mupdf.pdf_lexbuf_size_get, _mupdf.pdf_lexbuf_size_set)
    base_size = property(_mupdf.pdf_lexbuf_base_size_get, _mupdf.pdf_lexbuf_base_size_set)
    len = property(_mupdf.pdf_lexbuf_len_get, _mupdf.pdf_lexbuf_len_set)
    i = property(_mupdf.pdf_lexbuf_i_get, _mupdf.pdf_lexbuf_i_set)
    f = property(_mupdf.pdf_lexbuf_f_get, _mupdf.pdf_lexbuf_f_set)
    scratch = property(_mupdf.pdf_lexbuf_scratch_get, _mupdf.pdf_lexbuf_scratch_set)
    buffer = property(_mupdf.pdf_lexbuf_buffer_get, _mupdf.pdf_lexbuf_buffer_set)

    def __init__(self):
        _mupdf.pdf_lexbuf_swiginit(self, _mupdf.new_pdf_lexbuf())
    __swig_destroy__ = _mupdf.delete_pdf_lexbuf