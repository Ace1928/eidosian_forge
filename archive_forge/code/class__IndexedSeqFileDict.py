import os
import contextlib
import itertools
import collections.abc
from abc import ABC, abstractmethod
class _IndexedSeqFileDict(collections.abc.Mapping):
    """Read only dictionary interface to a sequential record file.

    This code is used in both Bio.SeqIO for indexing as SeqRecord
    objects, and in Bio.SearchIO for indexing QueryResult objects.

    Keeps the keys and associated file offsets in memory, reads the file
    to access entries as objects parsing them on demand. This approach
    is memory limited, but will work even with millions of records.

    Note duplicate keys are not allowed. If this happens, a ValueError
    exception is raised.

    As used in Bio.SeqIO, by default the SeqRecord's id string is used
    as the dictionary key. In Bio.SearchIO, the query's id string is
    used. This can be changed by supplying an optional key_function,
    a callback function which will be given the record id and must
    return the desired key. For example, this allows you to parse
    NCBI style FASTA identifiers, and extract the GI number to use
    as the dictionary key.

    Note that this dictionary is essentially read only. You cannot
    add or change values, pop values, nor clear the dictionary.
    """

    def __init__(self, random_access_proxy, key_function, repr, obj_repr):
        """Initialize the class."""
        self._proxy = random_access_proxy
        self._key_function = key_function
        self._repr = repr
        self._obj_repr = obj_repr
        self._cached_prev_record = (None, None)
        if key_function:
            offset_iter = ((key_function(key), offset, length) for key, offset, length in random_access_proxy)
        else:
            offset_iter = random_access_proxy
        offsets = {}
        for key, offset, length in offset_iter:
            if key in offsets:
                self._proxy._handle.close()
                raise ValueError(f"Duplicate key '{key}'")
            else:
                offsets[key] = offset
        self._offsets = offsets

    def __repr__(self):
        """Return a string representation of the File object."""
        return self._repr

    def __str__(self):
        """Create a string representation of the File object."""
        if self:
            return f'{{{list(self.keys())[0]!r} : {self._obj_repr}(...), ...}}'
        else:
            return '{}'

    def __len__(self):
        """Return the number of records."""
        return len(self._offsets)

    def __iter__(self):
        """Iterate over the keys."""
        return iter(self._offsets)

    def __getitem__(self, key):
        """Return record for the specified key.

        As an optimization when repeatedly asked to look up the same record,
        the key and record are cached so that if the *same* record is
        requested next time, it can be returned without going to disk.
        """
        if key == self._cached_prev_record[0]:
            return self._cached_prev_record[1]
        record = self._proxy.get(self._offsets[key])
        if self._key_function:
            key2 = self._key_function(record.id)
        else:
            key2 = record.id
        if key != key2:
            raise ValueError(f'Key did not match ({key} vs {key2})')
        self._cached_prev_record = (key, record)
        return record

    def get_raw(self, key):
        """Return the raw record from the file as a bytes string.

        If the key is not found, a KeyError exception is raised.
        """
        return self._proxy.get_raw(self._offsets[key])

    def close(self):
        """Close the file handle being used to read the data.

        Once called, further use of the index won't work. The sole purpose
        of this method is to allow explicit handle closure - for example
        if you wish to delete the file, on Windows you must first close
        all open handles to that file.
        """
        self._proxy._handle.close()