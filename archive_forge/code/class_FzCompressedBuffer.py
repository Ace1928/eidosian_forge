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
class FzCompressedBuffer(object):
    """
    Wrapper class for struct `fz_compressed_buffer`.
    Buffers of compressed data; typically for the source data
    for images.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_compressed_buffer_size(self):
        """
        Class-aware wrapper for `::fz_compressed_buffer_size()`.
        	Return the storage size used for a buffer and its data.
        	Used in implementing store handling.

        	Never throws exceptions.
        """
        return _mupdf.FzCompressedBuffer_fz_compressed_buffer_size(self)

    def fz_open_compressed_buffer(self):
        """
        Class-aware wrapper for `::fz_open_compressed_buffer()`.
        	Open a stream to read the decompressed version of a buffer.
        """
        return _mupdf.FzCompressedBuffer_fz_open_compressed_buffer(self)

    def fz_open_image_decomp_stream_from_buffer(self, l2factor):
        """
        Class-aware wrapper for `::fz_open_image_decomp_stream_from_buffer()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_open_image_decomp_stream_from_buffer()` => `(fz_stream *, int l2factor)`

        	Open a stream to read the decompressed version of a buffer,
        	with optional log2 subsampling.

        	l2factor = NULL for no subsampling, or a pointer to an integer
        	containing the maximum log2 subsample factor acceptable (0 =
        	none, 1 = halve dimensions, 2 = quarter dimensions etc). If
        	non-NULL, then *l2factor will be updated on exit with the actual
        	log2 subsample factor achieved.
        """
        return _mupdf.FzCompressedBuffer_fz_open_image_decomp_stream_from_buffer(self, l2factor)

    def get_buffer(self):
        """ Returns wrapper class for fz_buffer *m_internal.buffer."""
        return _mupdf.FzCompressedBuffer_get_buffer(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_compressed_buffer()`.
        		Create a new, UNKNOWN format, compressed_buffer.


        |

        *Overload 2:*
         Copy constructor using `fz_keep_compressed_buffer()`.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_compressed_buffer`.
        """
        _mupdf.FzCompressedBuffer_swiginit(self, _mupdf.new_FzCompressedBuffer(*args))
    __swig_destroy__ = _mupdf.delete_FzCompressedBuffer

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzCompressedBuffer_m_internal_value(self)
    m_internal = property(_mupdf.FzCompressedBuffer_m_internal_get, _mupdf.FzCompressedBuffer_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzCompressedBuffer_s_num_instances_get, _mupdf.FzCompressedBuffer_s_num_instances_set)