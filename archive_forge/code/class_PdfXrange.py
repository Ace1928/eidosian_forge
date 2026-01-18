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
class PdfXrange(object):
    """ Wrapper class for struct `pdf_xrange`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_xrange`.
        """
        _mupdf.PdfXrange_swiginit(self, _mupdf.new_PdfXrange(*args))
    __swig_destroy__ = _mupdf.delete_PdfXrange

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfXrange_m_internal_value(self)
    m_internal = property(_mupdf.PdfXrange_m_internal_get, _mupdf.PdfXrange_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfXrange_s_num_instances_get, _mupdf.PdfXrange_s_num_instances_set)