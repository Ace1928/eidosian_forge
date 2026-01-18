from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
class DichloroethaneValidation(SmartsValidation):
    """Logs if 1,2-dichloroethane is present.

    This is provided as an example of how to subclass :class:`~molvs.validations.SmartsValidation` to check for the
    presence of a substructure.
    """
    level = logging.INFO
    smarts = '[Cl]-[#6]-[#6]-[Cl]'
    entire_fragment = True
    message = '1,2-Dichloroethane is present'