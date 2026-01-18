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
class pdf_clean_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    write = property(_mupdf.pdf_clean_options_write_get, _mupdf.pdf_clean_options_write_set)
    image = property(_mupdf.pdf_clean_options_image_get, _mupdf.pdf_clean_options_image_set)
    subset_fonts = property(_mupdf.pdf_clean_options_subset_fonts_get, _mupdf.pdf_clean_options_subset_fonts_set)

    def __init__(self):
        _mupdf.pdf_clean_options_swiginit(self, _mupdf.new_pdf_clean_options())
    __swig_destroy__ = _mupdf.delete_pdf_clean_options