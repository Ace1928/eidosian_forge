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
class PdfMarkBits(object):
    """ Wrapper class for struct `pdf_mark_bits`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_mark_bits_reset(self):
        """ Class-aware wrapper for `::pdf_mark_bits_reset()`."""
        return _mupdf.PdfMarkBits_pdf_mark_bits_reset(self)

    def pdf_mark_bits_set(self, obj):
        """ Class-aware wrapper for `::pdf_mark_bits_set()`."""
        return _mupdf.PdfMarkBits_pdf_mark_bits_set(self, obj)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_new_mark_bits()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_mark_bits`.
        """
        _mupdf.PdfMarkBits_swiginit(self, _mupdf.new_PdfMarkBits(*args))
    __swig_destroy__ = _mupdf.delete_PdfMarkBits

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfMarkBits_m_internal_value(self)
    m_internal = property(_mupdf.PdfMarkBits_m_internal_get, _mupdf.PdfMarkBits_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfMarkBits_s_num_instances_get, _mupdf.PdfMarkBits_s_num_instances_set)