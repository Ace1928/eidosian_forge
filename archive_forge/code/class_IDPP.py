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
class IDPP(Calculator):
    """Image dependent pair potential.

    See:
        Improved initial guess for minimum energy path calculations.
        Søren Smidstrup, Andreas Pedersen, Kurt Stokbro and Hannes Jónsson
        Chem. Phys. 140, 214106 (2014)
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, target, mic):
        Calculator.__init__(self)
        self.target = target
        self.mic = mic

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        P = atoms.get_positions()
        d = []
        D = []
        for p in P:
            Di = P - p
            if self.mic:
                Di, di = find_mic(Di, atoms.get_cell(), atoms.get_pbc())
            else:
                di = np.sqrt((Di ** 2).sum(1))
            d.append(di)
            D.append(Di)
        d = np.array(d)
        D = np.array(D)
        dd = d - self.target
        d.ravel()[::len(d) + 1] = 1
        d4 = d ** 4
        e = 0.5 * (dd ** 2 / d4).sum()
        f = -2 * ((dd * (1 - 2 * dd / d) / d ** 5)[..., np.newaxis] * D).sum(0)
        self.results = {'energy': e, 'forces': f}