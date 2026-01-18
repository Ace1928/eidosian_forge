import sys
from copy import deepcopy
from functools import partial
from operator import mul, truediv
def _violates_constraint(fitness):
    return not fitness.valid and fitness.constraint_violation is not None and (sum(fitness.constraint_violation) > 0)