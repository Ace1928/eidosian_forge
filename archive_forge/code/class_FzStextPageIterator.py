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
class FzStextPageIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, item):
        _mupdf.FzStextPageIterator_swiginit(self, _mupdf.new_FzStextPageIterator(item))

    def __increment__(self):
        return _mupdf.FzStextPageIterator___increment__(self)

    def __eq__(self, rhs):
        return _mupdf.FzStextPageIterator___eq__(self, rhs)

    def __ne__(self, rhs):
        return _mupdf.FzStextPageIterator___ne__(self, rhs)

    def __ref__(self):
        return _mupdf.FzStextPageIterator___ref__(self)

    def __deref__(self):
        return _mupdf.FzStextPageIterator___deref__(self)
    __swig_destroy__ = _mupdf.delete_FzStextPageIterator

    def i_transform(self):
        """ Returns m_internal.u.i.transform if m_internal->type is FZ_STEXT_BLOCK_IMAGE, else throws."""
        return _mupdf.FzStextPageIterator_i_transform(self)

    def i_image(self):
        """ Returns m_internal.u.i.image if m_internal->type is FZ_STEXT_BLOCK_IMAGE, else throws."""
        return _mupdf.FzStextPageIterator_i_image(self)

    def begin(self):
        """ Used for iteration over linked list of FzStextLine items starting at fz_stext_line::u.t.first_line."""
        return _mupdf.FzStextPageIterator_begin(self)

    def end(self):
        """ Used for iteration over linked list of FzStextLine items starting at fz_stext_line::u.t.first_line."""
        return _mupdf.FzStextPageIterator_end(self)

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStextPageIterator_m_internal_value(self)
    m_internal = property(_mupdf.FzStextPageIterator_m_internal_get, _mupdf.FzStextPageIterator_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStextPageIterator_s_num_instances_get, _mupdf.FzStextPageIterator_s_num_instances_set)