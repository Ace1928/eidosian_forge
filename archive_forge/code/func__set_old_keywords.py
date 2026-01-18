import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def _set_old_keywords(self):
    """Store keywords for backwards compatibility wd VASP calculator"""
    self.spinpol = self.get_spin_polarized()
    self.energy_free = self.get_potential_energy(force_consistent=True)
    self.energy_zero = self.get_potential_energy(force_consistent=False)
    self.forces = self.get_forces()
    self.fermi = self.get_fermi_level()
    self.dipole = self.get_dipole_moment()
    self.stress = self.get_property('stress', allow_calculation=False)
    self.nbands = self.get_number_of_bands()