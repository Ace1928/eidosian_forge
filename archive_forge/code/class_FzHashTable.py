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
class FzHashTable(object):
    """
    Wrapper class for struct `fz_hash_table`. Not copyable or assignable.
    Generic hash-table with fixed-length keys.

    The keys and values are NOT reference counted by the hash table.
    Callers are responsible for taking care the reference counts are
    correct. Inserting a duplicate entry will NOT overwrite the old
    value, and will return the old value.

    The drop_val callback function is only used to release values
    when the hash table is destroyed.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_hash_filter(self, state, callback):
        """
        Class-aware wrapper for `::fz_hash_filter()`.
        	Iterate over the entries in a hash table, removing all the ones where callback returns true.
        	Does NOT free the value of the entry, so the caller is expected to take care of this.
        """
        return _mupdf.FzHashTable_fz_hash_filter(self, state, callback)

    def fz_hash_find(self, key):
        """
        Class-aware wrapper for `::fz_hash_find()`.
        	Search for a matching hash within the table, and return the
        	associated value.
        """
        return _mupdf.FzHashTable_fz_hash_find(self, key)

    def fz_hash_for_each(self, state, callback):
        """
        Class-aware wrapper for `::fz_hash_for_each()`.
        	Iterate over the entries in a hash table.
        """
        return _mupdf.FzHashTable_fz_hash_for_each(self, state, callback)

    def fz_hash_insert(self, key, val):
        """
        Class-aware wrapper for `::fz_hash_insert()`.
        	Insert a new key/value pair into the hash table.

        	If an existing entry with the same key is found, no change is
        	made to the hash table, and a pointer to the existing value is
        	returned.

        	If no existing entry with the same key is found, ownership of
        	val passes in, key is copied, and NULL is returned.
        """
        return _mupdf.FzHashTable_fz_hash_insert(self, key, val)

    def fz_hash_remove(self, key):
        """
        Class-aware wrapper for `::fz_hash_remove()`.
        	Remove the entry for a given key.

        	The value is NOT freed, so the caller is expected to take care
        	of this.
        """
        return _mupdf.FzHashTable_fz_hash_remove(self, key)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_hash_table()`.
        		Create a new hash table.

        		initialsize: The initial size of the hashtable. The hashtable
        		may grow (double in size) if it starts to get crowded (80%
        		full).

        		keylen: byte length for each key.

        		lock: -1 for no lock, otherwise the FZ_LOCK to use to protect
        		this table.

        		drop_val: Function to use to destroy values on table drop.


        |

        *Overload 2:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_hash_table`.
        """
        _mupdf.FzHashTable_swiginit(self, _mupdf.new_FzHashTable(*args))
    __swig_destroy__ = _mupdf.delete_FzHashTable

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzHashTable_m_internal_value(self)
    m_internal = property(_mupdf.FzHashTable_m_internal_get, _mupdf.FzHashTable_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzHashTable_s_num_instances_get, _mupdf.FzHashTable_s_num_instances_set)