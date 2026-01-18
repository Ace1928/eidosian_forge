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
def _fmax_all(self, images):
    """Store maximum force acting on each image in list. This is used in
           the dynamic optimization routine in the set_positions() function."""
    n = self.natoms
    forces = self.get_forces()
    fmax_images = [np.sqrt((forces[n * i:n + n * i] ** 2).sum(axis=1)).max() for i in range(self.nimages - 2)]
    return fmax_images