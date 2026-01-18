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
class FullSpringMethod(NEBMethod):
    """
    Elastic band method. The full spring force is included.
    """

    def get_tangent(self, state, spring1, spring2, i):
        tangent = spring1.t / spring1.nt + spring2.t / spring2.nt
        tangent /= np.linalg.norm(tangent)
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce, spring1, spring2, i):
        imgforce -= tangential_force * tangent
        energies = state.energies
        f1 = -(spring1.nt - state.eqlength) * spring1.t / spring1.nt * spring1.k
        f2 = (spring2.nt - state.eqlength) * spring2.t / spring2.nt * spring2.k
        if self.neb.climb and abs(i - self.neb.imax) == 1:
            deltavmax = max(abs(energies[i + 1] - energies[i]), abs(energies[i - 1] - energies[i]))
            deltavmin = min(abs(energies[i + 1] - energies[i]), abs(energies[i - 1] - energies[i]))
            imgforce += (f1 + f2) * deltavmin / deltavmax
        else:
            imgforce += f1 + f2