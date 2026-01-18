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
class FzOutlineIterator(object):
    """ Wrapper class for struct `fz_outline_iterator`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_load_outline_from_iterator(self):
        """
        Class-aware wrapper for `::fz_load_outline_from_iterator()`.
        	Routine to implement the old Structure based API from an iterator.
        """
        return _mupdf.FzOutlineIterator_fz_load_outline_from_iterator(self)

    def fz_outline_iterator_delete(self):
        """
        Class-aware wrapper for `::fz_outline_iterator_delete()`.
        	Delete the current item.

        	This implicitly moves us to the 'next' item, and the return code is as for fz_outline_iterator_next.
        """
        return _mupdf.FzOutlineIterator_fz_outline_iterator_delete(self)

    def fz_outline_iterator_down(self):
        """ Class-aware wrapper for `::fz_outline_iterator_down()`."""
        return _mupdf.FzOutlineIterator_fz_outline_iterator_down(self)

    def fz_outline_iterator_item(self):
        """
        Class-aware wrapper for `::fz_outline_iterator_item()`.
        	Call to get the current outline item.

        	Can return NULL. The item is only valid until the next call.
        """
        return _mupdf.FzOutlineIterator_fz_outline_iterator_item(self)

    def fz_outline_iterator_next(self):
        """
        Class-aware wrapper for `::fz_outline_iterator_next()`.
        	Calls to move the iterator position.

        	A negative return value means we could not move as requested. Otherwise:
        	0 = the final position has a valid item.
        	1 = not a valid item, but we can insert an item here.
        """
        return _mupdf.FzOutlineIterator_fz_outline_iterator_next(self)

    def fz_outline_iterator_prev(self):
        """ Class-aware wrapper for `::fz_outline_iterator_prev()`."""
        return _mupdf.FzOutlineIterator_fz_outline_iterator_prev(self)

    def fz_outline_iterator_up(self):
        """ Class-aware wrapper for `::fz_outline_iterator_up()`."""
        return _mupdf.FzOutlineIterator_fz_outline_iterator_up(self)

    def fz_outline_iterator_insert(self, item):
        """ Custom wrapper for fz_outline_iterator_insert()."""
        return _mupdf.FzOutlineIterator_fz_outline_iterator_insert(self, item)

    def fz_outline_iterator_update(self, item):
        """ Custom wrapper for fz_outline_iterator_update()."""
        return _mupdf.FzOutlineIterator_fz_outline_iterator_update(self, item)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_outline_iterator()`.
        		Get an iterator for the document outline.

        		Should be freed by fz_drop_outline_iterator.


        |

        *Overload 2:*
         Constructor using `fz_new_outline_iterator_of_size()`.

        |

        *Overload 3:*
         Constructor using `pdf_new_outline_iterator()`.

        |

        *Overload 4:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 5:*
         Constructor using raw copy of pre-existing `::fz_outline_iterator`.
        """
        _mupdf.FzOutlineIterator_swiginit(self, _mupdf.new_FzOutlineIterator(*args))
    __swig_destroy__ = _mupdf.delete_FzOutlineIterator

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzOutlineIterator_m_internal_value(self)
    m_internal = property(_mupdf.FzOutlineIterator_m_internal_get, _mupdf.FzOutlineIterator_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzOutlineIterator_s_num_instances_get, _mupdf.FzOutlineIterator_s_num_instances_set)