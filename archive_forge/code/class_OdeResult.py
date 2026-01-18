import inspect
import numpy as np
from .bdf import BDF
from .radau import Radau
from .rk import RK23, RK45, DOP853
from .lsoda import LSODA
from scipy.optimize import OptimizeResult
from .common import EPS, OdeSolution
from .base import OdeSolver
class OdeResult(OptimizeResult):
    pass