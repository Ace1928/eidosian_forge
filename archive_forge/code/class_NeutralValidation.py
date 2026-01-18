from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
class NeutralValidation(Validation):
    """Logs if not an overall neutral system."""

    def run(self, mol):
        charge = Chem.GetFormalCharge(mol)
        if not charge == 0:
            chargestring = '+%s' % charge if charge > 0 else '%s' % charge
            self.log.info(f'Not an overall neutral system ({chargestring})')