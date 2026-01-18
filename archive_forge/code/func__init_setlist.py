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
def _init_setlist(self):
    _datasets = dict()
    tmp_m = _data(**{'package': StrSexpVector((self._packagename,)), 'lib.loc': self._lib_loc})[2]
    nrows, ncols = tmp_m.do_slot('dim')
    c_i = 2
    for r_i in range(nrows):
        _datasets[tmp_m[r_i + c_i * nrows]] = None
    self._datasets = _datasets