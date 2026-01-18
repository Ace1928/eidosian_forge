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
def _guess_nimages(self):
    """Attempts to guess the number of images per band from
        a trajectory, based solely on the repetition of the
        potential energy of images. This should also work for symmetric
        cases."""
    e_first = self.images[0].get_potential_energy()
    nimages = None
    for index, image in enumerate(self.images[1:], start=1):
        e = image.get_potential_energy()
        if e == e_first:
            try:
                e_next = self.images[index + 1].get_potential_energy()
            except IndexError:
                pass
            else:
                if e_next == e_first:
                    nimages = index + 1
                    break
            nimages = index
            break
    if nimages is None:
        sys.stdout.write('Appears to be only one band in the images.\n')
        return len(self.images)
    e_last = self.images[nimages - 1].get_potential_energy()
    e_nextlast = self.images[2 * nimages - 1].get_potential_energy()
    if not e_last == e_nextlast:
        raise RuntimeError('Could not guess number of images per band.')
    sys.stdout.write('Number of images per band guessed to be {:d}.\n'.format(nimages))
    return nimages