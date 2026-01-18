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
class pdf_unsaved_sig(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    field = property(_mupdf.pdf_unsaved_sig_field_get, _mupdf.pdf_unsaved_sig_field_set)
    byte_range_start = property(_mupdf.pdf_unsaved_sig_byte_range_start_get, _mupdf.pdf_unsaved_sig_byte_range_start_set)
    byte_range_end = property(_mupdf.pdf_unsaved_sig_byte_range_end_get, _mupdf.pdf_unsaved_sig_byte_range_end_set)
    contents_start = property(_mupdf.pdf_unsaved_sig_contents_start_get, _mupdf.pdf_unsaved_sig_contents_start_set)
    contents_end = property(_mupdf.pdf_unsaved_sig_contents_end_get, _mupdf.pdf_unsaved_sig_contents_end_set)
    signer = property(_mupdf.pdf_unsaved_sig_signer_get, _mupdf.pdf_unsaved_sig_signer_set)
    next = property(_mupdf.pdf_unsaved_sig_next_get, _mupdf.pdf_unsaved_sig_next_set)

    def __init__(self):
        _mupdf.pdf_unsaved_sig_swiginit(self, _mupdf.new_pdf_unsaved_sig())
    __swig_destroy__ = _mupdf.delete_pdf_unsaved_sig