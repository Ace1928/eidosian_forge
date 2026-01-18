import collections
import logging
from operator import attrgetter
from pyomo.common.config import (
from pyomo.common.dependencies import scipy, numpy as np
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
@WriterFactory.register('compile_standard_form', 'Compile an LP to standard form (`min cTx s.t. Ax <= b`)')
class LinearStandardFormCompiler(object):
    CONFIG = ConfigBlock('compile_standard_form')
    CONFIG.declare('nonnegative_vars', ConfigValue(default=False, domain=bool, description='Convert all variables to be nonnegative variables'))
    CONFIG.declare('slack_form', ConfigValue(default=False, domain=bool, description='Add slack variables and return `min cTx s.t. Ax == b`'))
    CONFIG.declare('show_section_timing', ConfigValue(default=False, domain=bool, description='Print timing after each stage of the compilation process'))
    CONFIG.declare('file_determinism', ConfigValue(default=FileDeterminism.ORDERED, domain=InEnum(FileDeterminism), description='How much effort to ensure result is deterministic', doc='\n            How much effort do we want to put into ensuring the\n            resulting matrices are produced deterministically:\n                NONE (0) : None\n                ORDERED (10): rely on underlying component ordering (default)\n                SORT_INDICES (20) : sort keys of indexed components\n                SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)\n            '))
    CONFIG.declare('row_order', ConfigValue(default=None, description='Preferred constraint ordering', doc='\n            List of constraints in the order that they should appear in\n            the resulting `A` matrix.  Unspecified constraints will\n            appear at the end.'))
    CONFIG.declare('column_order', ConfigValue(default=None, description='Preferred variable ordering', doc='\n            List of variables in the order that they should appear in\n            the compiled representation.  Unspecified variables will be\n            appended to the end of this list.'))

    def __init__(self):
        self.config = self.CONFIG()

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream=None, **options):
        """Convert a model to standard form (`min cTx s.t. Ax <= b`)

        Returns
        -------
        LinearStandardFormInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: None
            This is provided for API compatibility with other writers
            and is ignored here.

        """
        config = self.config(options)
        with PauseGC():
            return _LinearStandardFormCompiler_impl(config).write(model)