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
class WeakPackage(Package):
    """
    'Weak' R package, with which looking for symbols results in
    a warning (and a None returned) whenever the desired symbol is
    not found (rather than a traditional `AttributeError`).
    """

    def __getattr__(self, name):
        res = self.__dict__.get(name)
        if res is None:
            warnings.warn("The symbol '%s' is not in this R namespace/package." % name)
        return res