from pyomo.core.expr.numeric_expr import LinearExpression
import pyomo.environ as pyo
from pyomo.core import Objective
An iterator to give representative Vars subject to non-anticipitivity
    Args:
        ef (ConcreteModel): the full extensive form model

    Yields:
        tree node name, full EF Var name, Var value
    