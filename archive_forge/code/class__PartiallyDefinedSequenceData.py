import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
class _PartiallyDefinedSequenceData(SequenceDataAbstractBaseClass):
    """Stores the length of a sequence with an undefined sequence contents (PRIVATE).

    Objects of this class can be used to create a Seq object to represent
    sequences with a known length, but with a sequence contents that is only
    partially known.
    Calling __len__ returns the sequence length, calling __getitem__ returns
    the sequence contents if known, otherwise an UndefinedSequenceError is
    raised.
    """
    __slots__ = ('_length', '_data')

    def __init__(self, length, data):
        """Initialize with the sequence length and defined sequence segments.

        The calling function is responsible for ensuring that the length is
        greater than zero.
        """
        self._length = length
        self._data = data
        super().__init__()

    def __getitem__(self, key: Union[slice, int]) -> Union[bytes, SequenceDataAbstractBaseClass]:
        if isinstance(key, slice):
            start, end, step = key.indices(self._length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
            data = {}
            for s, d in self._data.items():
                indices = range(-s, -s + self._length)[key]
                e: Optional[int] = indices.stop
                assert e is not None
                if step > 0:
                    if e <= 0:
                        continue
                    if indices.start < 0:
                        s = indices.start % step
                    else:
                        s = indices.start
                else:
                    if e < 0:
                        e = None
                    end = len(d) - 1
                    if indices.start > end:
                        s = end + (indices.start - end) % step
                    else:
                        s = indices.start
                    if s < 0:
                        continue
                start = (s - indices.start) // step
                d = d[s:e:step]
                if d:
                    data[start] = d
            if len(data) == 0:
                return _UndefinedSequenceData(size)
            end = -1
            previous = 0
            items = data.items()
            data = {}
            for start, seq in items:
                if end == start:
                    data[previous] += seq
                else:
                    data[start] = seq
                    previous = start
                end = start + len(seq)
            if len(data) == 1:
                seq = data.get(0)
                if seq is not None and len(seq) == size:
                    return seq
            if step < 0:
                data = {start: data[start] for start in reversed(list(data.keys()))}
            return _PartiallyDefinedSequenceData(size, data)
        elif self._length <= key:
            raise IndexError('sequence index out of range')
        else:
            for start, seq in self._data.items():
                if start <= key and key < start + len(seq):
                    return seq[key - start]
            raise UndefinedSequenceError('Sequence at position %d is undefined' % key)

    def __len__(self):
        return self._length

    def __bytes__(self):
        raise UndefinedSequenceError('Sequence content is only partially defined')

    def __add__(self, other):
        length = len(self) + len(other)
        data = dict(self._data)
        items = list(self._data.items())
        start, seq = items[-1]
        end = start + len(seq)
        try:
            other = bytes(other)
        except UndefinedSequenceError:
            if isinstance(other, _UndefinedSequenceData):
                pass
            elif isinstance(other, _PartiallyDefinedSequenceData):
                other_items = list(other._data.items())
                if end == len(self):
                    other_start, other_seq = other_items.pop(0)
                    if other_start == 0:
                        data[start] += other_seq
                    else:
                        data[len(self) + other_start] = other_seq
                for other_start, other_seq in other_items:
                    data[len(self) + other_start] = other_seq
        else:
            if end == len(self):
                data[start] += other
            else:
                data[len(self)] = other
        return _PartiallyDefinedSequenceData(length, data)

    def __radd__(self, other):
        length = len(other) + len(self)
        try:
            other = bytes(other)
        except UndefinedSequenceError:
            data = {len(other) + start: seq for start, seq in self._data.items()}
        else:
            data = {0: other}
            items = list(self._data.items())
            start, seq = items.pop(0)
            if start == 0:
                data[0] += seq
            else:
                data[len(other) + start] = seq
            for start, seq in items:
                data[len(other) + start] = seq
        return _PartiallyDefinedSequenceData(length, data)

    def __mul__(self, other):
        length = self._length
        items = self._data.items()
        data = {}
        end = -1
        previous = 0
        for i in range(other):
            for start, seq in items:
                start += i * length
                if end == start:
                    data[previous] += seq
                else:
                    data[start] = seq
                    previous = start
            end = start + len(seq)
        return _PartiallyDefinedSequenceData(length * other, data)

    def upper(self):
        """Return an upper case copy of the sequence."""
        data = {start: seq.upper() for start, seq in self._data.items()}
        return _PartiallyDefinedSequenceData(self._length, data)

    def lower(self):
        """Return a lower case copy of the sequence."""
        data = {start: seq.lower() for start, seq in self._data.items()}
        return _PartiallyDefinedSequenceData(self._length, data)

    def isupper(self):
        """Return True if all ASCII characters in data are uppercase.

        If there are no cased characters, the method returns False.
        """
        raise UndefinedSequenceError('Sequence content is only partially defined')

    def islower(self):
        """Return True if all ASCII characters in data are lowercase.

        If there are no cased characters, the method returns False.
        """
        raise UndefinedSequenceError('Sequence content is only partially defined')

    def translate(self, table, delete=b''):
        """Return a copy with each character mapped by the given translation table.

          table
            Translation table, which must be a bytes object of length 256.

        All characters occurring in the optional argument delete are removed.
        The remaining characters are mapped through the given translation table.
        """
        items = self._data.items()
        data = {start: seq.translate(table, delete) for start, seq in items}
        return _PartiallyDefinedSequenceData(self._length, data)

    def replace(self, old, new):
        """Return a copy with all occurrences of substring old replaced by new."""
        if len(old) != len(new):
            raise UndefinedSequenceError('Sequence content is only partially defined; substring \nreplacement cannot be performed reliably')
        items = self._data.items()
        data = {start: seq.replace(old, new) for start, seq in items}
        return _PartiallyDefinedSequenceData(self._length, data)

    @property
    def defined(self):
        """Return False, as the sequence is not fully defined and has a non-zero length."""
        return False

    @property
    def defined_ranges(self):
        """Return a tuple of the ranges where the sequence contents is defined.

        The return value has the format ((start1, end1), (start2, end2), ...).
        """
        return tuple(((start, start + len(seq)) for start, seq in self._data.items()))