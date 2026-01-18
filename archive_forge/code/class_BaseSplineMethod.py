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
class BaseSplineMethod(NEBMethod):
    """
    Base class for SplineNEB and String methods

    Can optionally be preconditioned, as described in the following article:

        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465
    """

    def __init__(self, neb):
        NEBMethod.__init__(self, neb)

    def get_tangent(self, state, spring1, spring2, i):
        return state.precon.get_tangent(i)

    def add_image_force(self, state, tangential_force, tangent, imgforce, spring1, spring2, i):
        imgforce -= tangential_force * tangent