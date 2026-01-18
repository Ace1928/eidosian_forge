from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict
class IndexConformanceException(Exception):
    pass