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
class pdf_rev_page_map(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    page = property(_mupdf.pdf_rev_page_map_page_get, _mupdf.pdf_rev_page_map_page_set)
    object = property(_mupdf.pdf_rev_page_map_object_get, _mupdf.pdf_rev_page_map_object_set)

    def __init__(self):
        _mupdf.pdf_rev_page_map_swiginit(self, _mupdf.new_pdf_rev_page_map())
    __swig_destroy__ = _mupdf.delete_pdf_rev_page_map