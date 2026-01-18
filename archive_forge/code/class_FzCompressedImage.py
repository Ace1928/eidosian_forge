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
class FzCompressedImage(object):
    """ Wrapper class for struct `fz_compressed_image`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_set_compressed_image_buffer(self, buf):
        """ Class-aware wrapper for `::fz_set_compressed_image_buffer()`."""
        return _mupdf.FzCompressedImage_fz_set_compressed_image_buffer(self, buf)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_compressed_image`.
        """
        _mupdf.FzCompressedImage_swiginit(self, _mupdf.new_FzCompressedImage(*args))
    __swig_destroy__ = _mupdf.delete_FzCompressedImage

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzCompressedImage_m_internal_value(self)
    m_internal = property(_mupdf.FzCompressedImage_m_internal_get, _mupdf.FzCompressedImage_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzCompressedImage_s_num_instances_get, _mupdf.FzCompressedImage_s_num_instances_set)