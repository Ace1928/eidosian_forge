import operator
from itertools import permutations, product
import pyomo.common.unittest as unittest
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import (
def _generate_possible_truth_inputs(nargs):
    return product([True, False], repeat=nargs)