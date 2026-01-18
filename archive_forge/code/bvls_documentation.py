import numpy as np
from numpy.linalg import norm, lstsq
from scipy.optimize import OptimizeResult
from .common import print_header_linear, print_iteration_linear
Compute the maximum violation of KKT conditions.