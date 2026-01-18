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
class MutableSeq(_SeqAbstractBaseClass):
    """An editable sequence object.

    Unlike normal python strings and our basic sequence object (the Seq class)
    which are immutable, the MutableSeq lets you edit the sequence in place.
    However, this means you cannot use a MutableSeq object as a dictionary key.

    >>> from Bio.Seq import MutableSeq
    >>> my_seq = MutableSeq("ACTCGTCGTCG")
    >>> my_seq
    MutableSeq('ACTCGTCGTCG')
    >>> my_seq[5]
    'T'
    >>> my_seq[5] = "A"
    >>> my_seq
    MutableSeq('ACTCGACGTCG')
    >>> my_seq[5]
    'A'
    >>> my_seq[5:8] = "NNN"
    >>> my_seq
    MutableSeq('ACTCGNNNTCG')
    >>> len(my_seq)
    11

    Note that the MutableSeq object does not support as many string-like
    or biological methods as the Seq object.
    """

    def __init__(self, data):
        """Create a MutableSeq object."""
        if isinstance(data, bytearray):
            self._data = data
        elif isinstance(data, bytes):
            self._data = bytearray(data)
        elif isinstance(data, str):
            self._data = bytearray(data, 'ASCII')
        elif isinstance(data, MutableSeq):
            self._data = data._data[:]
        elif isinstance(data, Seq):
            self._data = bytearray(bytes(data))
        else:
            raise TypeError('data should be a string, bytearray object, Seq object, or a MutableSeq object')

    def __setitem__(self, index, value):
        """Set a subsequence of single letter via value parameter.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> my_seq[0] = 'T'
        >>> my_seq
        MutableSeq('TCTCGACGTCG')
        """
        if isinstance(index, numbers.Integral):
            self._data[index] = ord(value)
        elif isinstance(value, MutableSeq):
            self._data[index] = value._data
        elif isinstance(value, Seq):
            self._data[index] = bytes(value)
        elif isinstance(value, str):
            self._data[index] = value.encode('ASCII')
        else:
            raise TypeError(f"received unexpected type '{type(value).__name__}'")

    def __delitem__(self, index):
        """Delete a subsequence of single letter.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> del my_seq[0]
        >>> my_seq
        MutableSeq('CTCGACGTCG')
        """
        del self._data[index]

    def append(self, c):
        """Add a subsequence to the mutable sequence object.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> my_seq.append('A')
        >>> my_seq
        MutableSeq('ACTCGACGTCGA')

        No return value.
        """
        self._data.append(ord(c.encode('ASCII')))

    def insert(self, i, c):
        """Add a subsequence to the mutable sequence object at a given index.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> my_seq.insert(0,'A')
        >>> my_seq
        MutableSeq('AACTCGACGTCG')
        >>> my_seq.insert(8,'G')
        >>> my_seq
        MutableSeq('AACTCGACGGTCG')

        No return value.
        """
        self._data.insert(i, ord(c.encode('ASCII')))

    def pop(self, i=-1):
        """Remove a subsequence of a single letter at given index.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> my_seq.pop()
        'G'
        >>> my_seq
        MutableSeq('ACTCGACGTC')
        >>> my_seq.pop()
        'C'
        >>> my_seq
        MutableSeq('ACTCGACGT')

        Returns the last character of the sequence.
        """
        c = self._data[i]
        del self._data[i]
        return chr(c)

    def remove(self, item):
        """Remove a subsequence of a single letter from mutable sequence.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> my_seq.remove('C')
        >>> my_seq
        MutableSeq('ATCGACGTCG')
        >>> my_seq.remove('A')
        >>> my_seq
        MutableSeq('TCGACGTCG')

        No return value.
        """
        codepoint = ord(item)
        try:
            self._data.remove(codepoint)
        except ValueError:
            raise ValueError('value not found in MutableSeq') from None

    def reverse(self):
        """Modify the mutable sequence to reverse itself.

        No return value.
        """
        self._data.reverse()

    def extend(self, other):
        """Add a sequence to the original mutable sequence object.

        >>> my_seq = MutableSeq('ACTCGACGTCG')
        >>> my_seq.extend('A')
        >>> my_seq
        MutableSeq('ACTCGACGTCGA')
        >>> my_seq.extend('TTT')
        >>> my_seq
        MutableSeq('ACTCGACGTCGATTT')

        No return value.
        """
        if isinstance(other, MutableSeq):
            self._data.extend(other._data)
        elif isinstance(other, Seq):
            self._data.extend(bytes(other))
        elif isinstance(other, str):
            self._data.extend(other.encode('ASCII'))
        else:
            raise TypeError('expected a string, Seq or MutableSeq')