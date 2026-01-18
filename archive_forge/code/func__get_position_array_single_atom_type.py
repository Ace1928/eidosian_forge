import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
def _get_position_array_single_atom_type(self, number):
    return self.atoms.get_scaled_positions()[self.atoms.numbers == number]