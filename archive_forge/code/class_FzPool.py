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
class FzPool(object):
    """
    Wrapper class for struct `fz_pool`. Not copyable or assignable.
    Simple pool allocators.

    Allocate from the pool, which can then be freed at once.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_pool_alloc(self, size):
        """
        Class-aware wrapper for `::fz_pool_alloc()`.
        	Allocate a block of size bytes from the pool.
        """
        return _mupdf.FzPool_fz_pool_alloc(self, size)

    def fz_pool_size(self):
        """
        Class-aware wrapper for `::fz_pool_size()`.
        	The current size of the pool.

        	The number of bytes of storage currently allocated to the pool.
        	This is the total of the storage used for the blocks making
        	up the pool, rather then total of the allocated blocks so far,
        	so it will increase in 'lumps'.
        	from the pool, then the pool size may still be X
        """
        return _mupdf.FzPool_fz_pool_size(self)

    def fz_pool_strdup(self, s):
        """
        Class-aware wrapper for `::fz_pool_strdup()`.
        	strdup equivalent allocating from the pool.
        """
        return _mupdf.FzPool_fz_pool_strdup(self, s)

    def fz_xml_add_att(self, node, key, val):
        """
        Class-aware wrapper for `::fz_xml_add_att()`.
        	Add an attribute to an XML node.
        """
        return _mupdf.FzPool_fz_xml_add_att(self, node, key, val)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_pool()`.
        		Create a new pool to allocate from.


        |

        *Overload 2:*
         Constructor using raw copy of pre-existing `::fz_pool`.
        """
        _mupdf.FzPool_swiginit(self, _mupdf.new_FzPool(*args))
    __swig_destroy__ = _mupdf.delete_FzPool

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPool_m_internal_value(self)
    m_internal = property(_mupdf.FzPool_m_internal_get, _mupdf.FzPool_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPool_s_num_instances_get, _mupdf.FzPool_s_num_instances_set)