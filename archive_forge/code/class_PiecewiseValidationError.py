import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
class PiecewiseValidationError(Exception):
    """An exception raised when validation of piecewise
    linear functions fail."""