import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r
class StringMethod(BaseSplineMethod):
    """
    String method using spline interpolation, plus optional preconditioning
    """

    def adjust_positions(self, positions):
        fit = self.neb.spline_fit(positions)
        new_s = np.linspace(0.0, 1.0, self.neb.nimages)
        new_positions = fit.x(new_s[1:-1]).reshape(-1, 3)
        return new_positions