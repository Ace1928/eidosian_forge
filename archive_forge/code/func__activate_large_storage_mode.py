import logging
import math
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.base import _no_ctype, _convert_ctype
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.container_utils import define_simple_containers
def _activate_large_storage_mode(self):
    if self.__byctype.__class__ is not dict:
        self_byctype = self.__dict__['_block__byctype'] = dict()
        for key, obj in self.__order.items():
            ctype = obj.ctype
            if ctype not in self_byctype:
                self_byctype[ctype] = dict()
            self_byctype[ctype][key] = obj