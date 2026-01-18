from typing import List
from pyomo.core.base.param import _ParamData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numvalue import value
from pyomo.contrib.appsi.base import PersistentBase
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.kernel.objective import minimize
from .config import WriterConfig
from pyomo.common.collections import OrderedSet
import os
from ..cmodel import cmodel, cmodel_available
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
def _set_pyomo_amplfunc_env(self):
    if self._external_functions:
        external_Libs = OrderedSet()
        for con, ext_funcs in self._external_functions.items():
            external_Libs.update([i._fcn._library for i in ext_funcs])
        set_pyomo_amplfunc_env(external_Libs)
    elif 'PYOMO_AMPLFUNC' in os.environ:
        del os.environ['PYOMO_AMPLFUNC']