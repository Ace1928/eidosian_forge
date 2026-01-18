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
class PdfFilterOptions(object):
    """ Wrapper class for struct `pdf_filter_options`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def add_factory(self, factory):
        """ We use default copy constructor and operator=.  Appends `factory` to internal vector and updates this->filters."""
        return _mupdf.PdfFilterOptions_add_factory(self, factory)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor initialises all fields to null/zero.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_filter_options`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_filter_options`.
        """
        _mupdf.PdfFilterOptions_swiginit(self, _mupdf.new_PdfFilterOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfFilterOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfFilterOptions
    recurse = property(_mupdf.PdfFilterOptions_recurse_get, _mupdf.PdfFilterOptions_recurse_set)
    instance_forms = property(_mupdf.PdfFilterOptions_instance_forms_get, _mupdf.PdfFilterOptions_instance_forms_set)
    ascii = property(_mupdf.PdfFilterOptions_ascii_get, _mupdf.PdfFilterOptions_ascii_set)
    no_update = property(_mupdf.PdfFilterOptions_no_update_get, _mupdf.PdfFilterOptions_no_update_set)
    opaque = property(_mupdf.PdfFilterOptions_opaque_get, _mupdf.PdfFilterOptions_opaque_set)
    complete = property(_mupdf.PdfFilterOptions_complete_get, _mupdf.PdfFilterOptions_complete_set)
    filters = property(_mupdf.PdfFilterOptions_filters_get, _mupdf.PdfFilterOptions_filters_set)
    newlines = property(_mupdf.PdfFilterOptions_newlines_get, _mupdf.PdfFilterOptions_newlines_set)
    s_num_instances = property(_mupdf.PdfFilterOptions_s_num_instances_get, _mupdf.PdfFilterOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfFilterOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfFilterOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfFilterOptions___ne__(self, rhs)
    m_filters = property(_mupdf.PdfFilterOptions_m_filters_get, _mupdf.PdfFilterOptions_m_filters_set)