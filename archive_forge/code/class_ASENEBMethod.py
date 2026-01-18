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
class ASENEBMethod(NEBMethod):
    """
    Standard NEB implementation in ASE. The tangent of each image is
    estimated from the spring closest to the saddle point in each
    spring pair.
    """

    def get_tangent(self, state, spring1, spring2, i):
        imax = self.neb.imax
        if i < imax:
            tangent = spring2.t
        elif i > imax:
            tangent = spring1.t
        else:
            tangent = spring1.t + spring2.t
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce, spring1, spring2, i):
        tangent_mag = np.vdot(tangent, tangent)
        factor = tangent / tangent_mag
        imgforce -= tangential_force * factor
        imgforce -= np.vdot(spring1.t * spring1.k - spring2.t * spring2.k, tangent) * factor