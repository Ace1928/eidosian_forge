from itertools import count
import numpy as np
from ase import Atoms
from ase.units import invcm, Ha
from ase.data import atomic_masses
from ase.calculators.calculator import all_changes
from ase.calculators.morse import MorsePotential
from ase.calculators.excitation_list import Excitation, ExcitationList
class H2Excitation(Excitation):

    def __eq__(self, other):
        """Considered to be equal when their indices are equal."""
        return self.index == other.index

    def __hash__(self):
        """Hash similar to __eq__"""
        if not hasattr(self, 'hash'):
            self.hash = hash(self.index)
        return self.hash