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
class PdfAlertEvent(object):
    """ Wrapper class for struct `pdf_alert_event`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_alert_event`.
        """
        _mupdf.PdfAlertEvent_swiginit(self, _mupdf.new_PdfAlertEvent(*args))
    __swig_destroy__ = _mupdf.delete_PdfAlertEvent

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfAlertEvent_m_internal_value(self)
    m_internal = property(_mupdf.PdfAlertEvent_m_internal_get, _mupdf.PdfAlertEvent_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfAlertEvent_s_num_instances_get, _mupdf.PdfAlertEvent_s_num_instances_set)