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
class FzLink(object):
    """
    Wrapper class for struct `fz_link`.
    fz_link is a list of interactive links on a page.

    There is no relation between the order of the links in the
    list and the order they appear on the page. The list of links
    for a given page can be obtained from fz_load_links.

    A link is reference counted. Dropping a reference to a link is
    done by calling fz_drop_link.

    rect: The hot zone. The area that can be clicked in
    untransformed coordinates.

    uri: Link destinations come in two forms: internal and external.
    Internal links refer to other pages in the same document.
    External links are URLs to other documents.

    next: A pointer to the next link on the same page.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_set_link_rect(self, rect):
        """ Class-aware wrapper for `::fz_set_link_rect()`."""
        return _mupdf.FzLink_fz_set_link_rect(self, rect)

    def fz_set_link_uri(self, uri):
        """ Class-aware wrapper for `::fz_set_link_uri()`."""
        return _mupdf.FzLink_fz_set_link_uri(self, uri)

    def begin(self):
        """ Used for iteration over linked list of FzLink items starting at fz_link::."""
        return _mupdf.FzLink_begin(self)

    def end(self):
        """ Used for iteration over linked list of FzLink items starting at fz_link::."""
        return _mupdf.FzLink_end(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_link_of_size()`.
        		Create a new link record.

        		next is set to NULL with the expectation that the caller will
        		handle the linked list setup. Internal function.

        		Different document types will be implemented by deriving from
        		fz_link. This macro allocates such derived structures, and
        		initialises the base sections.


        |

        *Overload 2:*
         Constructor using `pdf_new_link()`.

        |

        *Overload 3:*
         Construct by calling fz_new_link_of_size() with size=sizeof(fz_link).

        |

        *Overload 4:*
         Copy constructor using `fz_keep_link()`.

        |

        *Overload 5:*
         Constructor using raw copy of pre-existing `::fz_link`.

        |

        *Overload 6:*
         Constructor using raw copy of pre-existing `::fz_link`.
        """
        _mupdf.FzLink_swiginit(self, _mupdf.new_FzLink(*args))

    def refs(self):
        return _mupdf.FzLink_refs(self)

    def next(self):
        return _mupdf.FzLink_next(self)

    def rect(self):
        return _mupdf.FzLink_rect(self)

    def uri(self):
        return _mupdf.FzLink_uri(self)
    __swig_destroy__ = _mupdf.delete_FzLink

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzLink_m_internal_value(self)
    m_internal = property(_mupdf.FzLink_m_internal_get, _mupdf.FzLink_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzLink_s_num_instances_get, _mupdf.FzLink_s_num_instances_set)