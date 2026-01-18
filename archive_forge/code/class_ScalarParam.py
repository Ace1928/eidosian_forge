import sys
import types
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types, value as expr_value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.set import Reals, _AnySet, SetInitializer
from pyomo.core.base.units_container import units
from pyomo.core.expr import GetItemExpression
class ScalarParam(_ParamData, Param):

    def __init__(self, *args, **kwds):
        _ParamData.__init__(self, component=self)
        Param.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    def __call__(self, exception=True):
        """
        Return the value of this parameter.
        """
        if self._constructed:
            if not self._data:
                if self._mutable:
                    self[None]
                else:
                    return self[None]
            return super(ScalarParam, self).__call__(exception=exception)
        if exception:
            raise ValueError("Evaluating the numeric value of parameter '%s' before\n\tthe Param has been constructed (there is currently no value to return)." % (self.name,))

    def set_value(self, value, index=NOTSET):
        if index is NOTSET:
            index = None
        if self._constructed and (not self._mutable):
            _raise_modifying_immutable_error(self, index)
        if not self._data:
            self._data[index] = self
        super(ScalarParam, self).set_value(value, index)

    def is_constant(self):
        """Determine if this ScalarParam is constant (and can be eliminated)

        Returns False if either unconstructed or mutable, as it must be kept
        in expressions (as it either doesn't have a value yet or the value
        can change later.
        """
        return self._constructed and (not self._mutable)