import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def BoolFlag(val):
    return _true if val else _false