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
class ImprovedTangentMethod(NEBMethod):
    """
    Tangent estimates are improved according to Eqs. 8-11 in paper I.
    Tangents are weighted at extrema to ensure smooth transitions between
    the positive and negative tangents.
    """

    def get_tangent(self, state, spring1, spring2, i):
        energies = state.energies
        if energies[i + 1] > energies[i] > energies[i - 1]:
            tangent = spring2.t.copy()
        elif energies[i + 1] < energies[i] < energies[i - 1]:
            tangent = spring1.t.copy()
        else:
            deltavmax = max(abs(energies[i + 1] - energies[i]), abs(energies[i - 1] - energies[i]))
            deltavmin = min(abs(energies[i + 1] - energies[i]), abs(energies[i - 1] - energies[i]))
            if energies[i + 1] > energies[i - 1]:
                tangent = spring2.t * deltavmax + spring1.t * deltavmin
            else:
                tangent = spring2.t * deltavmin + spring1.t * deltavmax
        tangent /= np.linalg.norm(tangent)
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce, spring1, spring2, i):
        imgforce -= tangential_force * tangent
        imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent