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
class FzLocation(object):
    """
    Wrapper class for struct `fz_location`.
    Locations within the document are referred to in terms of
    chapter and page, rather than just a page number. For some
    documents (such as epub documents with large numbers of pages
    broken into many chapters) this can make navigation much faster
    as only the required chapter needs to be decoded at a time.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_make_location()`.
        		Simple constructor for fz_locations.


        |

        *Overload 2:*
         We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_location`.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_location`.
        """
        _mupdf.FzLocation_swiginit(self, _mupdf.new_FzLocation(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzLocation_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzLocation
    chapter = property(_mupdf.FzLocation_chapter_get, _mupdf.FzLocation_chapter_set)
    page = property(_mupdf.FzLocation_page_get, _mupdf.FzLocation_page_set)
    s_num_instances = property(_mupdf.FzLocation_s_num_instances_get, _mupdf.FzLocation_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzLocation_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzLocation___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzLocation___ne__(self, rhs)