import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
def __update_dict__(self, on_conflict='fail'):
    """ Update the __dict__ according to what is in the R environment """
    for elt in self._rpy2r:
        del self.__dict__[elt]
    self._rpy2r.clear()
    self.__fill_rpy2r__(on_conflict=on_conflict)