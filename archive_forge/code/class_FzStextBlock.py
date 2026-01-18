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
class FzStextBlock(object):
    """ Wrapper class for struct `fz_stext_block`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def i_transform(self):
        """ Returns m_internal.u.i.transform if m_internal->type is FZ_STEXT_BLOCK_IMAGE, else throws."""
        return _mupdf.FzStextBlock_i_transform(self)

    def i_image(self):
        """ Returns m_internal.u.i.image if m_internal->type is FZ_STEXT_BLOCK_IMAGE, else throws."""
        return _mupdf.FzStextBlock_i_image(self)

    def begin(self):
        """ Used for iteration over linked list of FzStextLine items starting at fz_stext_line::u.t.first_line."""
        return _mupdf.FzStextBlock_begin(self)

    def end(self):
        """ Used for iteration over linked list of FzStextLine items starting at fz_stext_line::u.t.first_line."""
        return _mupdf.FzStextBlock_end(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_stext_block`.
        """
        _mupdf.FzStextBlock_swiginit(self, _mupdf.new_FzStextBlock(*args))
    __swig_destroy__ = _mupdf.delete_FzStextBlock

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStextBlock_m_internal_value(self)
    m_internal = property(_mupdf.FzStextBlock_m_internal_get, _mupdf.FzStextBlock_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStextBlock_s_num_instances_get, _mupdf.FzStextBlock_s_num_instances_set)