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
class FzZipWriter(object):
    """
    Wrapper class for struct `fz_zip_writer`. Not copyable or assignable.
    fz_zip_writer offers methods for creating and writing zip files.
    It can be seen as the reverse of the fz_archive zip
    implementation.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_close_zip_writer(self):
        """
        Class-aware wrapper for `::fz_close_zip_writer()`.
        	Close the zip file for writing.

        	This flushes any pending data to the file. This can throw
        	exceptions.
        """
        return _mupdf.FzZipWriter_fz_close_zip_writer(self)

    def fz_write_zip_entry(self, name, buf, compress):
        """
        Class-aware wrapper for `::fz_write_zip_entry()`.
        	Given a buffer of data, (optionally) compress it, and add it to
        	the zip file with the given name.
        """
        return _mupdf.FzZipWriter_fz_write_zip_entry(self, name, buf, compress)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_zip_writer()`.
        		Create a new zip writer that writes to a given file.

        		Open an archive using a seekable stream object rather than
        		opening a file or directory on disk.


        |

        *Overload 2:*
         Constructor using `fz_new_zip_writer_with_output()`.
        		Create a new zip writer that writes to a given output stream.

        		Ownership of out passes in immediately upon calling this function.
        		The caller should never drop the fz_output, even if this function throws
        		an exception.


        |

        *Overload 3:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_zip_writer`.
        """
        _mupdf.FzZipWriter_swiginit(self, _mupdf.new_FzZipWriter(*args))
    __swig_destroy__ = _mupdf.delete_FzZipWriter

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzZipWriter_m_internal_value(self)
    m_internal = property(_mupdf.FzZipWriter_m_internal_get, _mupdf.FzZipWriter_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzZipWriter_s_num_instances_get, _mupdf.FzZipWriter_s_num_instances_set)