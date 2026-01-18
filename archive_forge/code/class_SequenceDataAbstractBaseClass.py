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
class SequenceDataAbstractBaseClass(ABC):
    """Abstract base class for sequence content providers.

    Most users will not need to use this class. It is used internally as a base
    class for sequence content provider classes such as _UndefinedSequenceData
    defined in this module, and _TwoBitSequenceData in Bio.SeqIO.TwoBitIO.
    Instances of these classes can be used instead of a ``bytes`` object as the
    data argument when creating a Seq object, and provide the sequence content
    only when requested via ``__getitem__``. This allows lazy parsers to load
    and parse sequence data from a file only for the requested sequence regions,
    and _UndefinedSequenceData instances to raise an exception when undefined
    sequence data are requested.

    Future implementations of lazy parsers that similarly provide on-demand
    parsing of sequence data should use a subclass of this abstract class and
    implement the abstract methods ``__len__`` and ``__getitem__``:

    * ``__len__`` must return the sequence length;
    * ``__getitem__`` must return

      * a ``bytes`` object for the requested region; or
      * a new instance of the subclass for the requested region; or
      * raise an ``UndefinedSequenceError``.

      Calling ``__getitem__`` for a sequence region of size zero should always
      return an empty ``bytes`` object.
      Calling ``__getitem__`` for the full sequence (as in data[:]) should
      either return a ``bytes`` object with the full sequence, or raise an
      ``UndefinedSequenceError``.

    Subclasses of SequenceDataAbstractBaseClass must call ``super().__init__()``
    as part of their ``__init__`` method.
    """
    __slots__ = ()

    def __init__(self):
        """Check if ``__getitem__`` returns a bytes-like object."""
        assert self[:0] == b''

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    def __bytes__(self):
        return self[:]

    def __hash__(self):
        return hash(bytes(self))

    def __eq__(self, other):
        return bytes(self) == other

    def __lt__(self, other):
        return bytes(self) < other

    def __le__(self, other):
        return bytes(self) <= other

    def __gt__(self, other):
        return bytes(self) > other

    def __ge__(self, other):
        return bytes(self) >= other

    def __add__(self, other):
        try:
            return bytes(self) + bytes(other)
        except UndefinedSequenceError:
            return NotImplemented

    def __radd__(self, other):
        return other + bytes(self)

    def __mul__(self, other):
        return other * bytes(self)

    def __contains__(self, item):
        return bytes(self).__contains__(item)

    def decode(self, encoding='utf-8'):
        """Decode the data as bytes using the codec registered for encoding.

        encoding
          The encoding with which to decode the bytes.
        """
        return bytes(self).decode(encoding)

    def count(self, sub, start=None, end=None):
        """Return the number of non-overlapping occurrences of sub in data[start:end].

        Optional arguments start and end are interpreted as in slice notation.
        This method behaves as the count method of Python strings.
        """
        return bytes(self).count(sub, start, end)

    def find(self, sub, start=None, end=None):
        """Return the lowest index in data where subsection sub is found.

        Return the lowest index in data where subsection sub is found,
        such that sub is contained within data[start,end].  Optional
        arguments start and end are interpreted as in slice notation.

        Return -1 on failure.
        """
        return bytes(self).find(sub, start, end)

    def rfind(self, sub, start=None, end=None):
        """Return the highest index in data where subsection sub is found.

        Return the highest index in data where subsection sub is found,
        such that sub is contained within data[start,end].  Optional
        arguments start and end are interpreted as in slice notation.

        Return -1 on failure.
        """
        return bytes(self).rfind(sub, start, end)

    def index(self, sub, start=None, end=None):
        """Return the lowest index in data where subsection sub is found.

        Return the lowest index in data where subsection sub is found,
        such that sub is contained within data[start,end].  Optional
        arguments start and end are interpreted as in slice notation.

        Raises ValueError when the subsection is not found.
        """
        return bytes(self).index(sub, start, end)

    def rindex(self, sub, start=None, end=None):
        """Return the highest index in data where subsection sub is found.

        Return the highest index in data where subsection sub is found,
        such that sub is contained within data[start,end].  Optional
        arguments start and end are interpreted as in slice notation.

        Raise ValueError when the subsection is not found.
        """
        return bytes(self).rindex(sub, start, end)

    def startswith(self, prefix, start=None, end=None):
        """Return True if data starts with the specified prefix, False otherwise.

        With optional start, test data beginning at that position.
        With optional end, stop comparing data at that position.
        prefix can also be a tuple of bytes to try.
        """
        return bytes(self).startswith(prefix, start, end)

    def endswith(self, suffix, start=None, end=None):
        """Return True if data ends with the specified suffix, False otherwise.

        With optional start, test data beginning at that position.
        With optional end, stop comparing data at that position.
        suffix can also be a tuple of bytes to try.
        """
        return bytes(self).endswith(suffix, start, end)

    def split(self, sep=None, maxsplit=-1):
        """Return a list of the sections in the data, using sep as the delimiter.

        sep
          The delimiter according which to split the data.
          None (the default value) means split on ASCII whitespace characters
          (space, tab, return, newline, formfeed, vertical tab).
        maxsplit
          Maximum number of splits to do.
          -1 (the default value) means no limit.
        """
        return bytes(self).split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        """Return a list of the sections in the data, using sep as the delimiter.

        sep
          The delimiter according which to split the data.
          None (the default value) means split on ASCII whitespace characters
          (space, tab, return, newline, formfeed, vertical tab).
        maxsplit
          Maximum number of splits to do.
          -1 (the default value) means no limit.

        Splitting is done starting at the end of the data and working to the front.
        """
        return bytes(self).rsplit(sep, maxsplit)

    def strip(self, chars=None):
        """Strip leading and trailing characters contained in the argument.

        If the argument is omitted or None, strip leading and trailing ASCII whitespace.
        """
        return bytes(self).strip(chars)

    def lstrip(self, chars=None):
        """Strip leading characters contained in the argument.

        If the argument is omitted or None, strip leading ASCII whitespace.
        """
        return bytes(self).lstrip(chars)

    def rstrip(self, chars=None):
        """Strip trailing characters contained in the argument.

        If the argument is omitted or None, strip trailing ASCII whitespace.
        """
        return bytes(self).rstrip(chars)

    def removeprefix(self, prefix):
        """Remove the prefix if present."""
        data = bytes(self)
        try:
            return data.removeprefix(prefix)
        except AttributeError:
            if data.startswith(prefix):
                return data[len(prefix):]
            else:
                return data

    def removesuffix(self, suffix):
        """Remove the suffix if present."""
        data = bytes(self)
        try:
            return data.removesuffix(suffix)
        except AttributeError:
            if data.startswith(suffix):
                return data[:-len(suffix)]
            else:
                return data

    def upper(self):
        """Return a copy of data with all ASCII characters converted to uppercase."""
        return bytes(self).upper()

    def lower(self):
        """Return a copy of data with all ASCII characters converted to lowercase."""
        return bytes(self).lower()

    def isupper(self):
        """Return True if all ASCII characters in data are uppercase.

        If there are no cased characters, the method returns False.
        """
        return bytes(self).isupper()

    def islower(self):
        """Return True if all ASCII characters in data are lowercase.

        If there are no cased characters, the method returns False.
        """
        return bytes(self).islower()

    def replace(self, old, new):
        """Return a copy with all occurrences of substring old replaced by new."""
        return bytes(self).replace(old, new)

    def translate(self, table, delete=b''):
        """Return a copy with each character mapped by the given translation table.

          table
            Translation table, which must be a bytes object of length 256.

        All characters occurring in the optional argument delete are removed.
        The remaining characters are mapped through the given translation table.
        """
        return bytes(self).translate(table, delete)

    @property
    def defined(self):
        """Return True if the sequence is defined, False if undefined or partially defined.

        Zero-length sequences are always considered to be defined.
        """
        return True

    @property
    def defined_ranges(self):
        """Return a tuple of the ranges where the sequence contents is defined.

        The return value has the format ((start1, end1), (start2, end2), ...).
        """
        length = len(self)
        if length > 0:
            return ((0, length),)
        else:
            return ()