from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
class DataShape(Mono):
    """
    Composite container for datashape elements.

    Elements of a datashape like ``Fixed(3)``, ``Var()`` or ``int32`` are on,
    on their own, valid datashapes.  These elements are collected together into
    a composite ``DataShape`` to be complete.

    This class is not intended to be used directly.  Instead, use the utility
    ``dshape`` function to create datashapes from strings or datashape
    elements.

    Examples
    --------

    >>> from datashader.datashape import Fixed, int32, DataShape, dshape

    >>> DataShape(Fixed(5), int32)  # Rare to DataShape directly
    dshape("5 * int32")

    >>> dshape('5 * int32')         # Instead use the dshape function
    dshape("5 * int32")

    >>> dshape([Fixed(5), int32])   # It can even do construction from elements
    dshape("5 * int32")

    See Also
    --------
    datashape.dshape
    """
    composite = False

    def __init__(self, *parameters, **kwds):
        if len(parameters) == 1 and isinstance(parameters[0], str):
            raise TypeError("DataShape constructor for internal use.\nUse dshape function to convert strings into datashapes.\nTry:\n\tdshape('%s')" % parameters[0])
        if len(parameters) > 0:
            self._parameters = tuple(map(_launder, parameters))
            if getattr(self._parameters[-1], 'cls', MEASURE) != MEASURE:
                raise TypeError('Only a measure can appear on the last position of a datashape, not %s' % repr(self._parameters[-1]))
            for dim in self._parameters[:-1]:
                if getattr(dim, 'cls', DIMENSION) != DIMENSION:
                    raise TypeError('Only dimensions can appear before the last position of a datashape, not %s' % repr(dim))
        else:
            raise ValueError('the data shape should be constructed from 2 or more parameters, only got %s' % len(parameters))
        self.composite = True
        self.name = kwds.get('name')
        if self.name:
            type(type(self))._registry[self.name] = self

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, index):
        return self.parameters[index]

    def __str__(self):
        return self.name or ' * '.join(map(str, self.parameters))

    def __repr__(self):
        s = pprint(self)
        if '\n' in s:
            return 'dshape("""%s""")' % s
        else:
            return 'dshape("%s")' % s

    @property
    def shape(self):
        return self.parameters[:-1]

    @property
    def measure(self):
        return self.parameters[-1]

    def subarray(self, leading):
        """Returns a data shape object of the subarray with 'leading'
        dimensions removed.

        >>> from datashader.datashape import dshape
        >>> dshape('1 * 2 * 3 * int32').subarray(1)
        dshape("2 * 3 * int32")
        >>> dshape('1 * 2 * 3 * int32').subarray(2)
        dshape("3 * int32")
        """
        if leading >= len(self.parameters):
            raise IndexError('Not enough dimensions in data shape to remove %d leading dimensions.' % leading)
        elif leading in [len(self.parameters) - 1, -1]:
            return DataShape(self.parameters[-1])
        else:
            return DataShape(*self.parameters[leading:])

    def __rmul__(self, other):
        if isinstance(other, int):
            other = Fixed(other)
        return DataShape(other, *self)

    @property
    def subshape(self):
        return IndexCallable(self._subshape)

    def _subshape(self, index):
        """ The DataShape of an indexed subarray

        >>> from datashader.datashape import dshape

        >>> ds = dshape('var * {name: string, amount: int32}')
        >>> print(ds.subshape[0])
        {name: string, amount: int32}

        >>> print(ds.subshape[0:3])
        3 * {name: string, amount: int32}

        >>> print(ds.subshape[0:7:2, 'amount'])
        4 * int32

        >>> print(ds.subshape[[1, 10, 15]])
        3 * {name: string, amount: int32}

        >>> ds = dshape('{x: int, y: int}')
        >>> print(ds.subshape['x'])
        int32

        >>> ds = dshape('10 * var * 10 * int32')
        >>> print(ds.subshape[0:5, 0:3, 5])
        5 * 3 * int32

        >>> ds = dshape('var * {name: string, amount: int32, id: int32}')
        >>> print(ds.subshape[:, [0, 2]])
        var * {name: string, id: int32}

        >>> ds = dshape('var * {name: string, amount: int32, id: int32}')
        >>> print(ds.subshape[:, ['name', 'id']])
        var * {name: string, id: int32}

        >>> print(ds.subshape[0, 1:])
        {amount: int32, id: int32}
        """
        from .predicates import isdimension
        if isinstance(index, int) and isdimension(self[0]):
            return self.subarray(1)
        if isinstance(self[0], Record) and isinstance(index, str):
            return self[0][index]
        if isinstance(self[0], Record) and isinstance(index, int):
            return self[0].parameters[0][index][1]
        if isinstance(self[0], Record) and isinstance(index, list):
            rec = self[0]
            index = [self[0].names.index(i) if isinstance(i, str) else i for i in index]
            return DataShape(Record([rec.parameters[0][i] for i in index]))
        if isinstance(self[0], Record) and isinstance(index, slice):
            rec = self[0]
            return DataShape(Record(rec.parameters[0][index]))
        if isinstance(index, list) and isdimension(self[0]):
            return len(index) * self.subarray(1)
        if isinstance(index, slice):
            if isinstance(self[0], Fixed):
                n = int(self[0])
                start = index.start or 0
                stop = index.stop or n
                if start < 0:
                    start = n + start
                if stop < 0:
                    stop = n + stop
                count = stop - start
            else:
                start = index.start or 0
                stop = index.stop
                if not stop:
                    count = -start if start < 0 else var
                if stop is not None and start is not None and (stop >= 0) and (start >= 0):
                    count = stop - start
                else:
                    count = var
            if count != var and index.step is not None:
                count = int(ceil(count / index.step))
            return count * self.subarray(1)
        if isinstance(index, tuple):
            if not index:
                return self
            elif index[0] is None:
                return 1 * self._subshape(index[1:])
            elif len(index) == 1:
                return self._subshape(index[0])
            else:
                ds = self.subarray(1)._subshape(index[1:])
                return (self[0] * ds)._subshape(index[0])
        raise TypeError('invalid index value %s of type %r' % (index, type(index).__name__))

    def __setstate__(self, state):
        self._parameters = state
        self.composite = True
        self.name = None