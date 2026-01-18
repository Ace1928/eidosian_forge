import os
import contextlib
import itertools
import collections.abc
from abc import ABC, abstractmethod
class _IndexedSeqFileProxy(ABC):
    """Abstract base class for file format specific random access (PRIVATE).

    This is subclasses in both Bio.SeqIO for indexing as SeqRecord
    objects, and in Bio.SearchIO for indexing QueryResult objects.

    Subclasses for each file format should define '__iter__', 'get'
    and optionally 'get_raw' methods.
    """

    @abstractmethod
    def __iter__(self):
        """Return (identifier, offset, length in bytes) tuples.

        The length can be zero where it is not implemented or not
        possible for a particular file format.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, offset):
        """Return parsed object for this entry."""
        raise NotImplementedError

    def get_raw(self, offset):
        """Return the raw record from the file as a bytes string (if implemented).

        If the key is not found, a KeyError exception is raised.

        This may not have been implemented for all file formats.
        """
        raise NotImplementedError('Not available for this file format.')