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
class pdf_font_resource_key(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    digest = property(_mupdf.pdf_font_resource_key_digest_get, _mupdf.pdf_font_resource_key_digest_set)
    type = property(_mupdf.pdf_font_resource_key_type_get, _mupdf.pdf_font_resource_key_type_set)
    encoding = property(_mupdf.pdf_font_resource_key_encoding_get, _mupdf.pdf_font_resource_key_encoding_set)
    local_xref = property(_mupdf.pdf_font_resource_key_local_xref_get, _mupdf.pdf_font_resource_key_local_xref_set)

    def __init__(self):
        _mupdf.pdf_font_resource_key_swiginit(self, _mupdf.new_pdf_font_resource_key())
    __swig_destroy__ = _mupdf.delete_pdf_font_resource_key