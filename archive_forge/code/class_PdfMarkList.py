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
class PdfMarkList(object):
    """ Wrapper class for struct `pdf_mark_list`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_mark_list_check(self, obj):
        """ Class-aware wrapper for `::pdf_mark_list_check()`."""
        return _mupdf.PdfMarkList_pdf_mark_list_check(self, obj)

    def pdf_mark_list_free(self):
        """ Class-aware wrapper for `::pdf_mark_list_free()`."""
        return _mupdf.PdfMarkList_pdf_mark_list_free(self)

    def pdf_mark_list_init(self):
        """ Class-aware wrapper for `::pdf_mark_list_init()`."""
        return _mupdf.PdfMarkList_pdf_mark_list_init(self)

    def pdf_mark_list_pop(self):
        """ Class-aware wrapper for `::pdf_mark_list_pop()`."""
        return _mupdf.PdfMarkList_pdf_mark_list_pop(self)

    def pdf_mark_list_push(self, obj):
        """ Class-aware wrapper for `::pdf_mark_list_push()`."""
        return _mupdf.PdfMarkList_pdf_mark_list_push(self, obj)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_mark_list`.
        """
        _mupdf.PdfMarkList_swiginit(self, _mupdf.new_PdfMarkList(*args))
    __swig_destroy__ = _mupdf.delete_PdfMarkList

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfMarkList_m_internal_value(self)
    m_internal = property(_mupdf.PdfMarkList_m_internal_get, _mupdf.PdfMarkList_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfMarkList_s_num_instances_get, _mupdf.PdfMarkList_s_num_instances_set)