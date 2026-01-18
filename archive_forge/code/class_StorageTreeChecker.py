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
class StorageTreeChecker(object):

    def __init__(self, model):
        self.tree = {model}
        self.model = model
        pb = self.parent_block(model)
        while pb is not None:
            self.tree.add(pb)
            pb = self.parent_block(pb)

    def __call__(self, comp, exception_flag=True):
        if comp is self.model:
            return True
        seen = set()
        pb = self.parent_block(comp)
        while pb is not None:
            if pb in self.tree:
                self.tree.update(seen)
                return True
            seen.add(pb)
            pb = self.parent_block(pb)
        if exception_flag:
            self.raise_error(comp)
        else:
            return False

    def parent_block(self, comp):
        if isinstance(comp, ICategorizedObject):
            parent = comp.parent
            while parent is not None and (not parent._is_heterogeneous_container):
                parent = parent.parent
            return parent
        else:
            return comp.parent_block()

    def raise_error(self, comp):
        raise RuntimeError("GAMS writer: found component '%s' not on same model tree.\nAll components must have the same parent model." % comp.name)