import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def _lagrange_legendre_transform_order2(v, s):
    ncp = s.get_discretization_info()['ncp']
    adotdot = s.get_discretization_info()['adotdot']

    def _fun(i):
        tmp = list(s)
        idx = s.ord(i) - 1
        if idx == 0:
            raise IndexError('list index out of range')
        elif i in s.get_finite_elements():
            raise IndexError('list index out of range')
        low = s.get_lower_element_boundary(i)
        lowidx = s.ord(low) - 1
        return sum((v(tmp[lowidx + j]) * adotdot[j][idx - lowidx] * (1.0 / (tmp[lowidx + ncp + 1] - tmp[lowidx]) ** 2) for j in range(ncp + 1)))
    return _fun