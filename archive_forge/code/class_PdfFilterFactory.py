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
class PdfFilterFactory(object):
    """ Wrapper class for struct `pdf_filter_factory`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_filter_factory`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_filter_factory`.
        """
        _mupdf.PdfFilterFactory_swiginit(self, _mupdf.new_PdfFilterFactory(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.PdfFilterFactory_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_PdfFilterFactory
    filter = property(_mupdf.PdfFilterFactory_filter_get, _mupdf.PdfFilterFactory_filter_set)
    options = property(_mupdf.PdfFilterFactory_options_get, _mupdf.PdfFilterFactory_options_set)
    s_num_instances = property(_mupdf.PdfFilterFactory_s_num_instances_get, _mupdf.PdfFilterFactory_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.PdfFilterFactory_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfFilterFactory___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.PdfFilterFactory___ne__(self, rhs)