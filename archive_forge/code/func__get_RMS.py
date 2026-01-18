import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
def _get_RMS(self, DW):
    if DW is None:
        if 'debye_waller_factors' in self.atoms.arrays:
            DW = {element: self._parse_array_from_atoms('debye_waller_factors', element, True) / (8 * np.pi ** 2) for element in self.atom_types}
    elif np.isscalar(DW):
        if len(self.atom_types) > 1:
            raise ValueError('This cell contains more then one type of atoms and the Debye-Waller factor needs to be provided for each atom using a dictionary.')
        DW = {self.atom_types[0]: DW / (8 * np.pi ** 2)}
    elif isinstance(DW, dict):
        verify_dictionary(self.atoms, DW, 'debye_waller_factors')
        for key, value in DW.items():
            DW[key] = value / (8 * np.pi ** 2)
    if DW is None:
        raise ValueError('Missing Debye-Waller factors. It can be provided as a dictionary with symbols as key or if the cell contains only a single type of element, the Debye-Waller factor can also be provided as float.')
    return DW