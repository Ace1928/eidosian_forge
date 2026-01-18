import logging
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component
from pyomo.common.collections import ComponentSet

    This function generates a ComponentSet of all Constraints, Expressions, and Objectives
    in a Block or model which have inconsistent units.

    Parameters
    ----------
    block : Pyomo Block or Model to test

    Returns
    ------
    ComponentSet : contains all Constraints, Expressions or Objectives which were
        identified as having unit consistency issues
    