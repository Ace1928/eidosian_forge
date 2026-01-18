import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class _DeferredUnitsSingleton(PyomoUnitsContainer):
    """A class supporting deferred interrogation of pint_available.

    This class supports creating a module-level singleton, but deferring
    the interrogation of the pint_available flag until the first time
    the object is actually used.  If pint is available, this instance
    object is replaced by an actual PyomoUnitsContainer.  Otherwise this
    leverages the pint_module to raise an (informative)
    DeferredImportError exception.

    """

    def __init__(self):
        pass

    def __getattribute__(self, attr):
        if attr == '__class__':
            return _DeferredUnitsSingleton
        if pint_available:
            if attr == 'set_pint_registry':
                pint_registry = None
            else:
                pint_registry = pint_module.UnitRegistry()
            self.__class__ = PyomoUnitsContainer
            self.__init__(pint_registry)
            return getattr(self, attr)
        else:
            return getattr(pint_module, attr)