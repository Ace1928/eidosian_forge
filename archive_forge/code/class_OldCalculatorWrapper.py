import warnings
from typing import Tuple
import numpy as np
from ase import __version__
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.constraints import dict2constraint
from ase.calculators.calculator import PropertyNotImplementedError
from ase.atoms import Atoms
from ase.io.jsonio import encode, decode
from ase.io.pickletrajectory import PickleTrajectory
from ase.parallel import world
from ase.utils import tokenize_version
class OldCalculatorWrapper:

    def __init__(self, calc):
        self.calc = calc
        try:
            self.name = calc.name
        except AttributeError:
            self.name = calc.__class__.__name__.lower()

    def get_property(self, prop, atoms, allow_calculation=True):
        try:
            if not allow_calculation and self.calc.calculation_required(atoms, [prop]):
                return None
        except AttributeError:
            pass
        method = 'get_' + {'energy': 'potential_energy', 'magmom': 'magnetic_moment', 'magmoms': 'magnetic_moments', 'dipole': 'dipole_moment'}.get(prop, prop)
        try:
            result = getattr(self.calc, method)(atoms)
        except AttributeError:
            raise PropertyNotImplementedError
        return result