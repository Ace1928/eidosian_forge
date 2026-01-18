from io import StringIO
from pyomo.common.gc_manager import PauseGC
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
import logging
class GAMSSymbolMap(SymbolMap):

    def __init__(self, var_labeler, var_list):
        super().__init__(self.var_label)
        self.var_labeler = var_labeler
        self.var_list = var_list

    def var_label(self, obj):
        return self.getSymbol(obj, self.var_recorder)

    def var_recorder(self, obj):
        ans = self.var_labeler(obj)
        try:
            if obj.is_variable_type():
                self.var_list.append(ans)
        except:
            pass
        return ans