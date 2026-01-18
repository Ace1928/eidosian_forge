import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
class OptimizerConvergenceError(Exception):
    pass