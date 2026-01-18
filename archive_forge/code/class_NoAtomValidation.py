from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
class NoAtomValidation(Validation):
    """Logs an error if the molecule has zero atoms.

    If the molecule has no atoms, no subsequent validations will run.
    """

    def run(self, mol):
        if mol.GetNumAtoms() == 0:
            self.log.error('No atoms are present')
            raise StopValidateError()