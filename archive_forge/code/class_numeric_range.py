import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
class numeric_range(abc.Sequence, abc.Hashable):
    """An extension of the built-in ``range()`` function whose arguments can
    be any orderable numeric type.

    With only *stop* specified, *start* defaults to ``0`` and *step*
    defaults to ``1``. The output items will match the type of *stop*:

        >>> list(numeric_range(3.5))
        [0.0, 1.0, 2.0, 3.0]

    With only *start* and *stop* specified, *step* defaults to ``1``. The
    output items will match the type of *start*:

        >>> from decimal import Decimal
        >>> start = Decimal('2.1')
        >>> stop = Decimal('5.1')
        >>> list(numeric_range(start, stop))
        [Decimal('2.1'), Decimal('3.1'), Decimal('4.1')]

    With *start*, *stop*, and *step*  specified the output items will match
    the type of ``start + step``:

        >>> from fractions import Fraction
        >>> start = Fraction(1, 2)  # Start at 1/2
        >>> stop = Fraction(5, 2)  # End at 5/2
        >>> step = Fraction(1, 2)  # Count by 1/2
        >>> list(numeric_range(start, stop, step))
        [Fraction(1, 2), Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)]

    If *step* is zero, ``ValueError`` is raised. Negative steps are supported:

        >>> list(numeric_range(3, -1, -1.0))
        [3.0, 2.0, 1.0, 0.0]

    Be aware of the limitations of floating point numbers; the representation
    of the yielded numbers may be surprising.

    ``datetime.datetime`` objects can be used for *start* and *stop*, if *step*
    is a ``datetime.timedelta`` object:

        >>> import datetime
        >>> start = datetime.datetime(2019, 1, 1)
        >>> stop = datetime.datetime(2019, 1, 3)
        >>> step = datetime.timedelta(days=1)
        >>> items = iter(numeric_range(start, stop, step))
        >>> next(items)
        datetime.datetime(2019, 1, 1, 0, 0)
        >>> next(items)
        datetime.datetime(2019, 1, 2, 0, 0)

    """
    _EMPTY_HASH = hash(range(0, 0))

    def __init__(self, *args):
        argc = len(args)
        if argc == 1:
            self._stop, = args
            self._start = type(self._stop)(0)
            self._step = type(self._stop - self._start)(1)
        elif argc == 2:
            self._start, self._stop = args
            self._step = type(self._stop - self._start)(1)
        elif argc == 3:
            self._start, self._stop, self._step = args
        elif argc == 0:
            raise TypeError('numeric_range expected at least 1 argument, got {}'.format(argc))
        else:
            raise TypeError('numeric_range expected at most 3 arguments, got {}'.format(argc))
        self._zero = type(self._step)(0)
        if self._step == self._zero:
            raise ValueError('numeric_range() arg 3 must not be zero')
        self._growing = self._step > self._zero
        self._init_len()

    def __bool__(self):
        if self._growing:
            return self._start < self._stop
        else:
            return self._start > self._stop

    def __contains__(self, elem):
        if self._growing:
            if self._start <= elem < self._stop:
                return (elem - self._start) % self._step == self._zero
        elif self._start >= elem > self._stop:
            return (self._start - elem) % -self._step == self._zero
        return False

    def __eq__(self, other):
        if isinstance(other, numeric_range):
            empty_self = not bool(self)
            empty_other = not bool(other)
            if empty_self or empty_other:
                return empty_self and empty_other
            else:
                return self._start == other._start and self._step == other._step and (self._get_by_index(-1) == other._get_by_index(-1))
        else:
            return False

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_by_index(key)
        elif isinstance(key, slice):
            step = self._step if key.step is None else key.step * self._step
            if key.start is None or key.start <= -self._len:
                start = self._start
            elif key.start >= self._len:
                start = self._stop
            else:
                start = self._get_by_index(key.start)
            if key.stop is None or key.stop >= self._len:
                stop = self._stop
            elif key.stop <= -self._len:
                stop = self._start
            else:
                stop = self._get_by_index(key.stop)
            return numeric_range(start, stop, step)
        else:
            raise TypeError('numeric range indices must be integers or slices, not {}'.format(type(key).__name__))

    def __hash__(self):
        if self:
            return hash((self._start, self._get_by_index(-1), self._step))
        else:
            return self._EMPTY_HASH

    def __iter__(self):
        values = (self._start + n * self._step for n in count())
        if self._growing:
            return takewhile(partial(gt, self._stop), values)
        else:
            return takewhile(partial(lt, self._stop), values)

    def __len__(self):
        return self._len

    def _init_len(self):
        if self._growing:
            start = self._start
            stop = self._stop
            step = self._step
        else:
            start = self._stop
            stop = self._start
            step = -self._step
        distance = stop - start
        if distance <= self._zero:
            self._len = 0
        else:
            q, r = divmod(distance, step)
            self._len = int(q) + int(r != self._zero)

    def __reduce__(self):
        return (numeric_range, (self._start, self._stop, self._step))

    def __repr__(self):
        if self._step == 1:
            return 'numeric_range({}, {})'.format(repr(self._start), repr(self._stop))
        else:
            return 'numeric_range({}, {}, {})'.format(repr(self._start), repr(self._stop), repr(self._step))

    def __reversed__(self):
        return iter(numeric_range(self._get_by_index(-1), self._start - self._step, -self._step))

    def count(self, value):
        return int(value in self)

    def index(self, value):
        if self._growing:
            if self._start <= value < self._stop:
                q, r = divmod(value - self._start, self._step)
                if r == self._zero:
                    return int(q)
        elif self._start >= value > self._stop:
            q, r = divmod(self._start - value, -self._step)
            if r == self._zero:
                return int(q)
        raise ValueError('{} is not in numeric range'.format(value))

    def _get_by_index(self, i):
        if i < 0:
            i += self._len
        if i < 0 or i >= self._len:
            raise IndexError('numeric range object index out of range')
        return self._start + i * self._step