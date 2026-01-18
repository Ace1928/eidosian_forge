import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.expression import _ExpressionData, _GeneralExpressionDataImpl
from pyomo.core.base.set import Set
from pyomo.core.base.initializer import (
from pyomo.core.base import minimize, maximize
@ModelComponentFactory.register('A list of objective expressions.')
class ObjectiveList(IndexedObjective):
    """
    An objective component that represents a list of objectives.
    Objectives can be indexed by their index, but when they are added
    an index value is not specified.
    """

    class End(object):
        pass

    def __init__(self, **kwargs):
        """Constructor"""
        if 'expr' in kwargs:
            raise ValueError("ObjectiveList does not accept the 'expr' keyword")
        _rule = kwargs.pop('rule', None)
        self._starting_index = kwargs.pop('starting_index', 1)
        super().__init__(Set(dimen=1), **kwargs)
        self.rule = Initializer(_rule, allow_generators=True)
        if self.rule is not None and type(self.rule) is IndexedCallInitializer:
            self.rule = CountedCallInitializer(self, self.rule, self._starting_index)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if self._constructed:
            return
        self._constructed = True
        if is_debug_set(logger):
            logger.debug('Constructing objective list %s' % self.name)
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()
        if self.rule is not None:
            _rule = self.rule(self.parent_block(), ())
            for cc in iter(_rule):
                if cc is ObjectiveList.End:
                    break
                if cc is Objective.Skip:
                    continue
                self.add(cc, sense=self._init_sense)

    def add(self, expr, sense=minimize):
        """Add an objective to the list."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        ans = self.__setitem__(next_idx, expr)
        if ans is not None:
            if sense not in {minimize, maximize}:
                sense = sense(self.parent_block(), next_idx)
            ans.set_sense(sense)
        return ans