from collections import namedtuple
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import ZeroConstant, native_numeric_types, as_numeric
from pyomo.core import Constraint, Var, Block, Set
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
import logging
@ModelComponentFactory.register('A list of complementarity conditions.')
class ComplementarityList(IndexedComplementarity):
    """
    A complementarity component that represents a list of complementarity
    conditions.  Each condition can be indexed by its index, but when added
    an index value is not specified.
    """
    End = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        args = (Set(),)
        self._nconditions = 0
        Complementarity.__init__(self, *args, **kwargs)
        self._rule = None

    def add(self, expr):
        """
        Add a complementarity condition with an implicit index.
        """
        self._nconditions += 1
        self._index_set.add(self._nconditions)
        return Complementarity.add(self, self._nconditions, expr)

    def construct(self, data=None):
        """
        Construct the expression(s) for this complementarity condition.
        """
        if self._constructed:
            return
        self._constructed = True
        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug('Constructing complementarity list %s', self.name)
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()
        if self._init_rule is not None:
            _init = self._init_rule(self.parent_block(), ())
            for cc in iter(_init):
                if cc is ComplementarityList.End:
                    break
                if cc is Complementarity.Skip:
                    continue
                self.add(cc)
        timer.report()