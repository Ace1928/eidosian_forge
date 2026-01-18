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
class pdf_page(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    super = property(_mupdf.pdf_page_super_get, _mupdf.pdf_page_super_set)
    doc = property(_mupdf.pdf_page_doc_get, _mupdf.pdf_page_doc_set)
    obj = property(_mupdf.pdf_page_obj_get, _mupdf.pdf_page_obj_set)
    transparency = property(_mupdf.pdf_page_transparency_get, _mupdf.pdf_page_transparency_set)
    overprint = property(_mupdf.pdf_page_overprint_get, _mupdf.pdf_page_overprint_set)
    links = property(_mupdf.pdf_page_links_get, _mupdf.pdf_page_links_set)
    annots = property(_mupdf.pdf_page_annots_get, _mupdf.pdf_page_annots_set)
    annot_tailp = property(_mupdf.pdf_page_annot_tailp_get, _mupdf.pdf_page_annot_tailp_set)
    widgets = property(_mupdf.pdf_page_widgets_get, _mupdf.pdf_page_widgets_set)
    widget_tailp = property(_mupdf.pdf_page_widget_tailp_get, _mupdf.pdf_page_widget_tailp_set)

    def __init__(self):
        _mupdf.pdf_page_swiginit(self, _mupdf.new_pdf_page())
    __swig_destroy__ = _mupdf.delete_pdf_page