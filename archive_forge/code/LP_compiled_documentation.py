import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
from pyomo.repn.beta.matrix import compile_block_linear_constraints

    A continuous linear model that is compiled into a
    MatrixConstraint object.
    