import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary, Integers
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

    A discrete model where discrete variables have custom bounds
    