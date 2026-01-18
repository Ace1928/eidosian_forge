import time
import logging
import array
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.expr.numvalue import is_fixed, ZeroConstant
from pyomo.core.base.set_types import Any
from pyomo.core.base import SortComponents, Var
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import (
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.repn import generate_standard_repn
from collections.abc import Mapping
@ModelComponentFactory.register('A set of constraint expressions in Ax=b form.')
class MatrixConstraint(Mapping, IndexedConstraint):
    StrictUpperBound = 3
    UpperBound = 2
    Equality = 14
    LowerBound = 8
    StrictLowerBound = 24
    NoBound = 0

    def __init__(self, nrows, ncols, nnz, prows, jcols, vals, ranges, range_types, varmap):
        assert len(prows) == nrows + 1
        assert len(jcols) == nnz
        assert len(vals) == nnz
        assert len(ranges) == 2 * nrows
        assert len(range_types) == nrows
        assert len(varmap) == ncols
        IndexedConstraint.__init__(self, Any)
        self._prows = prows
        self._jcols = jcols
        self._vals = vals
        self._ranges = ranges
        self._range_types = range_types
        self._varmap = varmap

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if is_debug_set(logger):
            logger.debug('Constructing constraint %s' % self.name)
        if self._constructed:
            return
        self._constructed = True
        _init = _LinearMatrixConstraintData
        self._data = tuple((_init(i, component=self) for i in range(len(self._range_types))))

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        return iter((i for i in range(len(self))))

    def add(self, index, expr):
        raise NotImplementedError

    def __delitem__(self):
        raise NotImplementedError

    def keys(self, sort=None):
        return super().keys()

    def values(self, sort=None):
        return super().values()

    def items(self, sort=None):
        return super().items()