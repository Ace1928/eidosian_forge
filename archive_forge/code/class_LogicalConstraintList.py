import inspect
import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.boolean_value import as_boolean, BooleanConstant
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set
@ModelComponentFactory.register('A list of logical constraints.')
class LogicalConstraintList(IndexedLogicalConstraint):
    """
    A logical constraint component that represents a list of constraints.
    Constraints can be indexed by their index, but when they are
    added an index value is not specified.
    """
    End = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        if 'expr' in kwargs:
            raise ValueError("LogicalConstraintList does not accept the 'expr' keyword")
        LogicalConstraint.__init__(self, Set(dimen=1), **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this logical constraint.
        """
        if self._constructed:
            return
        self._constructed = True
        generate_debug_messages = is_debug_set(logger)
        if generate_debug_messages:
            logger.debug('Constructing logical constraint list %s' % self.name)
        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()
        assert self._init_expr is None
        _init_rule = self.rule
        self._init_expr = None
        if _init_rule is None:
            return
        _generator = None
        _self_parent = self._parent()
        if inspect.isgeneratorfunction(_init_rule):
            _generator = _init_rule(_self_parent)
        elif inspect.isgenerator(_init_rule):
            _generator = _init_rule
        if _generator is None:
            while True:
                val = len(self._index_set) + 1
                if generate_debug_messages:
                    logger.debug('   Constructing logical constraint index ' + str(val))
                expr = apply_indexed_rule(self, _init_rule, _self_parent, val)
                if expr is None:
                    raise ValueError("LogicalConstraintList '%s': rule returned None instead of LogicalConstraintList.End" % (self.name,))
                if expr.__class__ is tuple and expr == LogicalConstraintList.End:
                    return
                self.add(expr)
        else:
            for expr in _generator:
                if expr is None:
                    raise ValueError("LogicalConstraintList '%s': generator returned None instead of LogicalConstraintList.End" % (self.name,))
                if expr.__class__ is tuple and expr == LogicalConstraintList.End:
                    return
                self.add(expr)

    def add(self, expr):
        """Add a logical constraint with an implicit index."""
        next_idx = len(self._index_set) + 1
        self._index_set.add(next_idx)
        return self.__setitem__(next_idx, expr)