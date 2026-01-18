import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
class NumpyNdIterType(IteratorType):
    """
    Type class for `np.nditer()` objects.

    The layout denotes in which order the logical shape is iterated on.
    "C" means logical order (corresponding to in-memory order in C arrays),
    "F" means reverse logical order (corresponding to in-memory order in
    F arrays).
    """

    def __init__(self, arrays):
        self.arrays = tuple(arrays)
        self.layout = self._compute_layout(self.arrays)
        self.dtypes = tuple((getattr(a, 'dtype', a) for a in self.arrays))
        self.ndim = max((getattr(a, 'ndim', 0) for a in self.arrays))
        name = 'nditer(ndim={ndim}, layout={layout}, inputs={arrays})'.format(ndim=self.ndim, layout=self.layout, arrays=self.arrays)
        super(NumpyNdIterType, self).__init__(name)

    @classmethod
    def _compute_layout(cls, arrays):
        c = collections.Counter()
        for a in arrays:
            if not isinstance(a, Array):
                continue
            if a.layout in 'CF' and a.ndim == 1:
                c['C'] += 1
                c['F'] += 1
            elif a.ndim >= 1:
                c[a.layout] += 1
        return 'F' if c['F'] > c['C'] else 'C'

    @property
    def key(self):
        return self.arrays

    @property
    def views(self):
        """
        The views yielded by the iterator.
        """
        return [Array(dtype, 0, 'C') for dtype in self.dtypes]

    @property
    def yield_type(self):
        from . import BaseTuple
        views = self.views
        if len(views) > 1:
            return BaseTuple.from_types(views)
        else:
            return views[0]

    @cached_property
    def indexers(self):
        """
        A list of (kind, start_dim, end_dim, indices) where:
        - `kind` is either "flat", "indexed", "0d" or "scalar"
        - `start_dim` and `end_dim` are the dimension numbers at which
          this indexing takes place
        - `indices` is the indices of the indexed arrays in self.arrays
        """
        d = collections.OrderedDict()
        layout = self.layout
        ndim = self.ndim
        assert layout in 'CF'
        for i, a in enumerate(self.arrays):
            if not isinstance(a, Array):
                indexer = ('scalar', 0, 0)
            elif a.ndim == 0:
                indexer = ('0d', 0, 0)
            else:
                if a.layout == layout or (a.ndim == 1 and a.layout in 'CF'):
                    kind = 'flat'
                else:
                    kind = 'indexed'
                if layout == 'C':
                    indexer = (kind, ndim - a.ndim, ndim)
                else:
                    indexer = (kind, 0, a.ndim)
            d.setdefault(indexer, []).append(i)
        return list((k + (v,) for k, v in d.items()))

    @cached_property
    def need_shaped_indexing(self):
        """
        Whether iterating on this iterator requires keeping track of
        individual indices inside the shape.  If False, only a single index
        over the equivalent flat shape is required, which can make the
        iterator more efficient.
        """
        for kind, start_dim, end_dim, _ in self.indexers:
            if kind in ('0d', 'scalar'):
                pass
            elif kind == 'flat':
                if (start_dim, end_dim) != (0, self.ndim):
                    return True
            else:
                return True
        return False