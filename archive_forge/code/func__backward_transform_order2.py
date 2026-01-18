import logging
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def _backward_transform_order2(v, s):
    """
    Applies the Backward Difference formula of order O(h) for second
    derivatives
    """

    def _bwd_fun(i):
        tmp = list(s)
        idx = s.ord(i) - 1
        if idx == 0 or idx == 1:
            raise IndexError('list index out of range')
        return 1 / ((tmp[idx - 1] - tmp[idx - 2]) * (tmp[idx] - tmp[idx - 1])) * (v(tmp[idx]) - 2 * v(tmp[idx - 1]) + v(tmp[idx - 2]))
    return _bwd_fun