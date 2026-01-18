import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.utils import reader
from .utils import verify_cell_for_export, verify_dictionary
def _get_occupancies(self):
    if 'occupancies' in self.atoms.arrays:
        occupancies = self.atoms.get_array('occupancies', copy=False)
    else:
        occupancies = np.ones_like(self.atoms.numbers)
    return occupancies