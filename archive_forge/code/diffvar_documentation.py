import weakref
from pyomo.common.collections import ComponentMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
Sets``_expr``, an expression representing the discretization
        equations linking the :class:`DerivativeVar` to its state
        :class:`Var`
        