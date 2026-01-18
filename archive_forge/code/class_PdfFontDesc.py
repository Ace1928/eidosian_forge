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
class PdfFontDesc(object):
    """ Wrapper class for struct `pdf_font_desc`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_add_hmtx(self, lo, hi, w):
        """ Class-aware wrapper for `::pdf_add_hmtx()`."""
        return _mupdf.PdfFontDesc_pdf_add_hmtx(self, lo, hi, w)

    def pdf_add_vmtx(self, lo, hi, x, y, w):
        """ Class-aware wrapper for `::pdf_add_vmtx()`."""
        return _mupdf.PdfFontDesc_pdf_add_vmtx(self, lo, hi, x, y, w)

    def pdf_end_hmtx(self):
        """ Class-aware wrapper for `::pdf_end_hmtx()`."""
        return _mupdf.PdfFontDesc_pdf_end_hmtx(self)

    def pdf_end_vmtx(self):
        """ Class-aware wrapper for `::pdf_end_vmtx()`."""
        return _mupdf.PdfFontDesc_pdf_end_vmtx(self)

    def pdf_font_cid_to_gid(self, cid):
        """ Class-aware wrapper for `::pdf_font_cid_to_gid()`."""
        return _mupdf.PdfFontDesc_pdf_font_cid_to_gid(self, cid)

    def pdf_set_default_hmtx(self, w):
        """ Class-aware wrapper for `::pdf_set_default_hmtx()`."""
        return _mupdf.PdfFontDesc_pdf_set_default_hmtx(self, w)

    def pdf_set_default_vmtx(self, y, w):
        """ Class-aware wrapper for `::pdf_set_default_vmtx()`."""
        return _mupdf.PdfFontDesc_pdf_set_default_vmtx(self, y, w)

    def pdf_set_font_wmode(self, wmode):
        """ Class-aware wrapper for `::pdf_set_font_wmode()`."""
        return _mupdf.PdfFontDesc_pdf_set_font_wmode(self, wmode)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_new_font_desc()`.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_font_desc`.
        """
        _mupdf.PdfFontDesc_swiginit(self, _mupdf.new_PdfFontDesc(*args))
    __swig_destroy__ = _mupdf.delete_PdfFontDesc

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfFontDesc_m_internal_value(self)
    m_internal = property(_mupdf.PdfFontDesc_m_internal_get, _mupdf.PdfFontDesc_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfFontDesc_s_num_instances_get, _mupdf.PdfFontDesc_s_num_instances_set)