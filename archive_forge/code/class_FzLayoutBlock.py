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
class FzLayoutBlock(object):
    """ Wrapper class for struct `fz_layout_block`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_add_layout_char(self, x, w, p):
        """
        Class-aware wrapper for `::fz_add_layout_char()`.
        	Add a new char to the line at the end of the layout block.
        """
        return _mupdf.FzLayoutBlock_fz_add_layout_char(self, x, w, p)

    def fz_add_layout_line(self, x, y, h, p):
        """
        Class-aware wrapper for `::fz_add_layout_line()`.
        	Add a new line to the end of the layout block.
        """
        return _mupdf.FzLayoutBlock_fz_add_layout_line(self, x, y, h, p)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_layout()`.
        		Create a new layout block, with new allocation pool, zero
        		matrices, and initialise linked pointers.


        |

        *Overload 2:*
         Constructor using raw copy of pre-existing `::fz_layout_block`.
        """
        _mupdf.FzLayoutBlock_swiginit(self, _mupdf.new_FzLayoutBlock(*args))
    __swig_destroy__ = _mupdf.delete_FzLayoutBlock

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzLayoutBlock_m_internal_value(self)
    m_internal = property(_mupdf.FzLayoutBlock_m_internal_get, _mupdf.FzLayoutBlock_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzLayoutBlock_s_num_instances_get, _mupdf.FzLayoutBlock_s_num_instances_set)