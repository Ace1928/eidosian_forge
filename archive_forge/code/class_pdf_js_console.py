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
class pdf_js_console(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    drop = property(_mupdf.pdf_js_console_drop_get, _mupdf.pdf_js_console_drop_set)
    show = property(_mupdf.pdf_js_console_show_get, _mupdf.pdf_js_console_show_set)
    hide = property(_mupdf.pdf_js_console_hide_get, _mupdf.pdf_js_console_hide_set)
    clear = property(_mupdf.pdf_js_console_clear_get, _mupdf.pdf_js_console_clear_set)
    write = property(_mupdf.pdf_js_console_write_get, _mupdf.pdf_js_console_write_set)

    def __init__(self):
        _mupdf.pdf_js_console_swiginit(self, _mupdf.new_pdf_js_console())
    __swig_destroy__ = _mupdf.delete_pdf_js_console