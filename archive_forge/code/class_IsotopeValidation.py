from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
class IsotopeValidation(Validation):
    """Logs if molecule contains isotopes."""

    def run(self, mol):
        isotopes = set()
        for atom in mol.GetAtoms():
            isotope = atom.GetIsotope()
            if not isotope == 0:
                isotopes.add(f'{isotope}{atom.GetSymbol()}')
        for isotope in isotopes:
            self.log.info(f'Molecule contains isotope {isotope}')