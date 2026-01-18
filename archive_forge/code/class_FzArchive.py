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
class FzArchive(object):
    """
    Wrapper class for struct `fz_archive`.
    fz_archive:

    fz_archive provides methods for accessing "archive" files.
    An archive file is a conceptual entity that contains multiple
    files, which can be counted, enumerated, and read.

    Implementations of fz_archive based upon directories, zip
    and tar files are included.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_archive_format(self):
        """
        Class-aware wrapper for `::fz_archive_format()`.
        	Return a pointer to a string describing the format of the
        	archive.

        	The lifetime of the string is unspecified (in current
        	implementations the string will persist until the archive
        	is closed, but this is not guaranteed).
        """
        return _mupdf.FzArchive_fz_archive_format(self)

    def fz_count_archive_entries(self):
        """
        Class-aware wrapper for `::fz_count_archive_entries()`.
        	Number of entries in archive.

        	Will always return a value >= 0.

        	May throw an exception if this type of archive cannot count the
        	entries (such as a directory).
        """
        return _mupdf.FzArchive_fz_count_archive_entries(self)

    def fz_has_archive_entry(self, name):
        """
        Class-aware wrapper for `::fz_has_archive_entry()`.
        	Check if entry by given name exists.

        	If named entry does not exist 0 will be returned, if it does
        	exist 1 is returned.

        	name: Entry name to look for, this must be an exact match to
        	the entry name in the archive.
        """
        return _mupdf.FzArchive_fz_has_archive_entry(self, name)

    def fz_list_archive_entry(self, idx):
        """
        Class-aware wrapper for `::fz_list_archive_entry()`.
        	Get listed name of entry position idx.

        	idx: Must be a value >= 0 < return value from
        	fz_count_archive_entries. If not in range NULL will be
        	returned.

        	May throw an exception if this type of archive cannot list the
        	entries (such as a directory).
        """
        return _mupdf.FzArchive_fz_list_archive_entry(self, idx)

    def fz_mount_multi_archive(self, sub, path):
        """
        Class-aware wrapper for `::fz_mount_multi_archive()`.
        	Add an archive to the set of archives handled by a multi
        	archive.

        	If path is NULL, then the archive contents will appear at the
        	top level, otherwise, the archives contents will appear prefixed
        	by path.
        """
        return _mupdf.FzArchive_fz_mount_multi_archive(self, sub, path)

    def fz_open_archive_entry(self, name):
        """
        Class-aware wrapper for `::fz_open_archive_entry()`.
        	Opens an archive entry as a stream.

        	name: Entry name to look for, this must be an exact match to
        	the entry name in the archive.

        	Throws an exception if a matching entry cannot be found.
        """
        return _mupdf.FzArchive_fz_open_archive_entry(self, name)

    def fz_parse_xml_archive_entry(self, filename, preserve_white):
        """
        Class-aware wrapper for `::fz_parse_xml_archive_entry()`.
        	Parse the contents of an archive entry into a tree of xml nodes.

        	preserve_white: whether to keep or delete all-whitespace nodes.
        """
        return _mupdf.FzArchive_fz_parse_xml_archive_entry(self, filename, preserve_white)

    def fz_read_archive_entry(self, name):
        """
        Class-aware wrapper for `::fz_read_archive_entry()`.
        	Reads all bytes in an archive entry
        	into a buffer.

        	name: Entry name to look for, this must be an exact match to
        	the entry name in the archive.

        	Throws an exception if a matching entry cannot be found.
        """
        return _mupdf.FzArchive_fz_read_archive_entry(self, name)

    def fz_tree_archive_add_buffer(self, name, buf):
        """
        Class-aware wrapper for `::fz_tree_archive_add_buffer()`.
        	Add a named buffer to an existing tree archive.

        	The tree will take a new reference to the buffer. Ownership
        	is not transferred.
        """
        return _mupdf.FzArchive_fz_tree_archive_add_buffer(self, name, buf)

    def fz_tree_archive_add_data(self, name, data, size):
        """
        Class-aware wrapper for `::fz_tree_archive_add_data()`.
        	Add a named block of data to an existing tree archive.

        	The data will be copied into a buffer, and so the caller
        	may free it as soon as this returns.
        """
        return _mupdf.FzArchive_fz_tree_archive_add_data(self, name, data, size)

    def fz_try_open_archive_entry(self, name):
        """
        Class-aware wrapper for `::fz_try_open_archive_entry()`.
        	Opens an archive entry as a stream.

        	Returns NULL if a matching entry cannot be found, otherwise
        	behaves exactly as fz_open_archive_entry.
        """
        return _mupdf.FzArchive_fz_try_open_archive_entry(self, name)

    def fz_try_parse_xml_archive_entry(self, filename, preserve_white):
        """
        Class-aware wrapper for `::fz_try_parse_xml_archive_entry()`.
        	Try and parse the contents of an archive entry into a tree of xml nodes.

        	preserve_white: whether to keep or delete all-whitespace nodes.

        	Will return NULL if the archive entry can't be found. Otherwise behaves
        	the same as fz_parse_xml_archive_entry. May throw exceptions.
        """
        return _mupdf.FzArchive_fz_try_parse_xml_archive_entry(self, filename, preserve_white)

    def fz_try_read_archive_entry(self, name):
        """
        Class-aware wrapper for `::fz_try_read_archive_entry()`.
        	Reads all bytes in an archive entry
        	into a buffer.

        	name: Entry name to look for, this must be an exact match to
        	the entry name in the archive.

        	Returns NULL if a matching entry cannot be found. Otherwise behaves
        	the same as fz_read_archive_entry. Exceptions may be thrown.
        """
        return _mupdf.FzArchive_fz_try_read_archive_entry(self, name)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_archive_of_size()`.

        |

        *Overload 2:*
         Constructor using `fz_new_multi_archive()`.
        		Create a new multi archive (initially empty).


        |

        *Overload 3:*
         Constructor using `fz_new_tree_archive()`.
        		Create an archive that holds named buffers.

        		tree can either be a preformed tree with fz_buffers as values,
        		or it can be NULL for an empty tree.


        |

        *Overload 4:*
         Copy constructor using `fz_keep_archive()`.

        |

        *Overload 5:*
         Constructor using raw copy of pre-existing `::fz_archive`.
        """
        _mupdf.FzArchive_swiginit(self, _mupdf.new_FzArchive(*args))
    __swig_destroy__ = _mupdf.delete_FzArchive

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzArchive_m_internal_value(self)
    m_internal = property(_mupdf.FzArchive_m_internal_get, _mupdf.FzArchive_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzArchive_s_num_instances_get, _mupdf.FzArchive_s_num_instances_set)