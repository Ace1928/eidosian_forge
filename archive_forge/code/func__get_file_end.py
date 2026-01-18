import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
def _get_file_end(self):
    return 'Orientation\n   1 0 0\n   0 1 0\n   0 0 1\n'