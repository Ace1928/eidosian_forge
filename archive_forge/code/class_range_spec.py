class range_spec(object):
    """A single contiguous (byte) range.

    A range_spec defines a range (of bytes) by specifying two offsets,
    the 'first' and 'last', which are inclusive in the range.  Offsets
    are zero-based (the first byte is offset 0).  The range can not be
    empty or negative (has to satisfy first <= last).

    The range can be unbounded on either end, represented here by the
    None value, with these semantics:

       * A 'last' of None always indicates the last possible byte
        (although that offset may not be known).

       * A 'first' of None indicates this is a suffix range, where
         the last value is actually interpreted to be the number
         of bytes at the end of the file (regardless of file size).

    Note that it is not valid for both first and last to be None.

    """
    __slots__ = ['first', 'last']

    def __init__(self, first=0, last=None):
        self.set(first, last)

    def set(self, first, last):
        """Sets the value of this range given the first and last offsets.
        """
        if first is not None and last is not None and (first > last):
            raise ValueError('Byte range does not satisfy first <= last.')
        elif first is None and last is None:
            raise ValueError('Byte range can not omit both first and last offsets.')
        self.first = first
        self.last = last

    def __repr__(self):
        return '%s.%s(%s,%s)' % (self.__class__.__module__, self.__class__.__name__, self.first, self.last)

    def __str__(self):
        """Returns a string form of the range as would appear in a Range: header."""
        if self.first is None and self.last is None:
            return ''
        s = ''
        if self.first is not None:
            s += '%d' % self.first
        s += '-'
        if self.last is not None:
            s += '%d' % self.last
        return s

    def __eq__(self, other):
        """Compare ranges for equality.

        Note that if non-specific ranges are involved (such as 34- and -5),
        they could compare as not equal even though they may represent
        the same set of bytes in some contexts.
        """
        return self.first == other.first and self.last == other.last

    def __ne__(self, other):
        """Compare ranges for inequality.

        Note that if non-specific ranges are involved (such as 34- and -5),
        they could compare as not equal even though they may represent
        the same set of bytes in some contexts.
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """< operator is not defined"""
        raise NotImplementedError('Ranges can not be relationally compared')

    def __le__(self, other):
        """<= operator is not defined"""
        raise NotImplementedError('Ranges can not be ralationally compared')

    def __gt__(self, other):
        """> operator is not defined"""
        raise NotImplementedError('Ranges can not be relationally compared')

    def __ge__(self, other):
        """>= operator is not defined"""
        raise NotImplementedError('Ranges can not be relationally compared')

    def copy(self):
        """Makes a copy of this range object."""
        return self.__class__(self.first, self.last)

    def is_suffix(self):
        """Returns True if this is a suffix range.

        A suffix range is one that specifies the last N bytes of a
        file regardless of file size.

        """
        return self.first == None

    def is_fixed(self):
        """Returns True if this range is absolute and a fixed size.

        This occurs only if neither first or last is None.  Converse
        is the is_unbounded() method.

        """
        return self.first is not None and self.last is not None

    def is_unbounded(self):
        """Returns True if the number of bytes in the range is unspecified.

        This can only occur if either the 'first' or the 'last' member
        is None.  Converse is the is_fixed() method.

        """
        return self.first is None or self.last is None

    def is_whole_file(self):
        """Returns True if this range includes all possible bytes.

        This can only occur if the 'last' member is None and the first
        member is 0.

        """
        return self.first == 0 and self.last is None

    def __contains__(self, offset):
        """Does this byte range contain the given byte offset?

        If the offset < 0, then it is taken as an offset from the end
        of the file, where -1 is the last byte.  This type of offset
        will only work with suffix ranges.

        """
        if offset < 0:
            if self.first is not None:
                return False
            else:
                return self.last >= -offset
        elif self.first is None:
            return False
        elif self.last is None:
            return True
        else:
            return self.first <= offset <= self.last

    def fix_to_size(self, size):
        """Changes a length-relative range to an absolute range based upon given file size.

        Ranges that are already absolute are left as is.

        Note that zero-length files are handled as special cases,
        since the only way possible to specify a zero-length range is
        with the suffix range "-0".  Thus unless this range is a suffix
        range, it can not satisfy a zero-length file.

        If the resulting range (partly) lies outside the file size then an
        error is raised.
        """
        if size == 0:
            if self.first is None:
                self.last = 0
                return
            else:
                raise RangeUnsatisfiableError('Range can satisfy a zero-length file.')
        if self.first is None:
            self.first = size - self.last
            if self.first < 0:
                self.first = 0
            self.last = size - 1
        elif self.first > size - 1:
            raise RangeUnsatisfiableError('Range begins beyond the file size.')
        elif self.last is None:
            self.last = size - 1
        return

    def merge_with(self, other):
        """Tries to merge the given range into this one.

        The size of this range may be enlarged as a result.

        An error is raised if the two ranges do not overlap or are not
        contiguous with each other.
        """
        if self.is_whole_file() or self == other:
            return
        elif other.is_whole_file():
            self.first, self.last = (0, None)
            return
        a1, z1 = (self.first, self.last)
        a2, z2 = (other.first, other.last)
        if self.is_suffix():
            if z1 == 0:
                self.first, self.last = (a2, z2)
                return
            elif other.is_suffix():
                self.last = max(z1, z2)
            else:
                raise RangeUnmergableError()
        elif other.is_suffix():
            if z2 == 0:
                return
            else:
                raise RangeUnmergableError()
        assert a1 is not None and a2 is not None
        if a2 < a1:
            a1, z1, a2, z2 = (a2, z2, a1, z1)
        assert a1 <= a2
        if z1 is None:
            if z2 is not None and z2 + 1 < a1:
                raise RangeUnmergableError()
            else:
                self.first = min(a1, a2)
                self.last = None
        elif z2 is None:
            if z1 + 1 < a2:
                raise RangeUnmergableError()
            else:
                self.first = min(a1, a2)
                self.last = None
        elif a2 > z1 + 1:
            raise RangeUnmergableError()
        else:
            self.first = a1
            self.last = max(z1, z2)
        return