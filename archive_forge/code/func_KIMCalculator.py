import re
import os
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammps import convert
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError
def KIMCalculator(model_name, options, debug):
    """
    Used only for Portable Models
    """
    options_not_allowed = ['modelname', 'debug']
    _check_conflict_options(options, options_not_allowed, simulator='kimmodel')
    return KIMModelCalculator(model_name, debug=debug, **options)