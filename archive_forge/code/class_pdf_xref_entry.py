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
class pdf_xref_entry(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    type = property(_mupdf.pdf_xref_entry_type_get, _mupdf.pdf_xref_entry_type_set)
    marked = property(_mupdf.pdf_xref_entry_marked_get, _mupdf.pdf_xref_entry_marked_set)
    gen = property(_mupdf.pdf_xref_entry_gen_get, _mupdf.pdf_xref_entry_gen_set)
    num = property(_mupdf.pdf_xref_entry_num_get, _mupdf.pdf_xref_entry_num_set)
    ofs = property(_mupdf.pdf_xref_entry_ofs_get, _mupdf.pdf_xref_entry_ofs_set)
    stm_ofs = property(_mupdf.pdf_xref_entry_stm_ofs_get, _mupdf.pdf_xref_entry_stm_ofs_set)
    stm_buf = property(_mupdf.pdf_xref_entry_stm_buf_get, _mupdf.pdf_xref_entry_stm_buf_set)
    obj = property(_mupdf.pdf_xref_entry_obj_get, _mupdf.pdf_xref_entry_obj_set)

    def __init__(self):
        _mupdf.pdf_xref_entry_swiginit(self, _mupdf.new_pdf_xref_entry())
    __swig_destroy__ = _mupdf.delete_pdf_xref_entry