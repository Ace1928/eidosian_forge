import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class ConstraintWrapper:
    """Object to wrap constraints to pass to `multiprocessing.Pool`."""

    def __init__(self, g_cons, g_cons_args):
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def gcons(self, v_x_a):
        vfeasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            if np.any(g(v_x_a, *args) < 0.0):
                vfeasible = False
                break
        return vfeasible