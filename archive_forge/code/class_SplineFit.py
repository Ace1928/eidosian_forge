import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
class SplineFit:
    """
    Fit a cubic spline fit to images
    """

    def __init__(self, s, x):
        self._s = s
        self._x_data = x
        self._x = CubicSpline(self._s, x, bc_type='not-a-knot')
        self._dx_ds = self._x.derivative()
        self._d2x_ds2 = self._x.derivative(2)

    @property
    def s(self):
        return self._s

    @property
    def x_data(self):
        return self._x_data

    @property
    def x(self):
        return self._x

    @property
    def dx_ds(self):
        return self._dx_ds

    @property
    def d2x_ds2(self):
        return self._d2x_ds2