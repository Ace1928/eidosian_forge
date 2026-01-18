from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
def A_rule(model, i, j):
    if i == 4:
        i = 5
    elif i == 5:
        i = 4
    return 2 if i == j else 1