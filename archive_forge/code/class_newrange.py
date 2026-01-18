from __future__ import absolute_import
from future.utils import PY2
from itertools import islice
from future.backports.misc import count   # with step parameter on Py2.6
class newrange(Sequence):
    """
    Pure-Python backport of Python 3's range object.  See `the CPython
    documentation for details:
    <http://docs.python.org/py3k/library/functions.html#range>`_
    """

    def __init__(self, *args):
        if len(args) == 1:
            start, stop, step = (0, args[0], 1)
        elif len(args) == 2:
            start, stop, step = (args[0], args[1], 1)
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise TypeError('range() requires 1-3 int arguments')
        try:
            start, stop, step = (int(start), int(stop), int(step))
        except ValueError:
            raise TypeError('an integer is required')
        if step == 0:
            raise ValueError('range() arg 3 must not be zero')
        elif step < 0:
            stop = min(stop, start)
        else:
            stop = max(stop, start)
        self._start = start
        self._stop = stop
        self._step = step
        self._len = (stop - start) // step + bool((stop - start) % step)

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def __repr__(self):
        if self._step == 1:
            return 'range(%d, %d)' % (self._start, self._stop)
        return 'range(%d, %d, %d)' % (self._start, self._stop, self._step)

    def __eq__(self, other):
        return isinstance(other, newrange) and (self._len == 0 == other._len or (self._start, self._step, self._len) == (other._start, other._step, other._len))

    def __len__(self):
        return self._len

    def index(self, value):
        """Return the 0-based position of integer `value` in
        the sequence this range represents."""
        try:
            diff = value - self._start
        except TypeError:
            raise ValueError('%r is not in range' % value)
        quotient, remainder = divmod(diff, self._step)
        if remainder == 0 and 0 <= quotient < self._len:
            return abs(quotient)
        raise ValueError('%r is not in range' % value)

    def count(self, value):
        """Return the number of occurrences of integer `value`
        in the sequence this range represents."""
        return int(value in self)

    def __contains__(self, value):
        """Return ``True`` if the integer `value` occurs in
        the sequence this range represents."""
        try:
            self.index(value)
            return True
        except ValueError:
            return False

    def __reversed__(self):
        return iter(self[::-1])

    def __getitem__(self, index):
        """Return the element at position ``index`` in the sequence
        this range represents, or raise :class:`IndexError` if the
        position is out of range."""
        if isinstance(index, slice):
            return self.__getitem_slice(index)
        if index < 0:
            index = self._len + index
        if index < 0 or index >= self._len:
            raise IndexError('range object index out of range')
        return self._start + index * self._step

    def __getitem_slice(self, slce):
        """Return a range which represents the requested slce
        of the sequence represented by this range.
        """
        scaled_indices = (self._step * n for n in slce.indices(self._len))
        start_offset, stop_offset, new_step = scaled_indices
        return newrange(self._start + start_offset, self._start + stop_offset, new_step)

    def __iter__(self):
        """Return an iterator which enumerates the elements of the
        sequence this range represents."""
        return range_iterator(self)