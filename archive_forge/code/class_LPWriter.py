import logging
from io import StringIO
from operator import itemgetter, attrgetter
from pyomo.common.config import (
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.label import LPFileLabeler, NumericLabeler
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
@WriterFactory.register('cpxlp_v2', 'Generate the corresponding CPLEX LP file (version 2).')
@WriterFactory.register('lp_v2', 'Generate the corresponding LP file (version 2).')
class LPWriter(object):
    CONFIG = ConfigBlock('lpwriter')
    CONFIG.declare('show_section_timing', ConfigValue(default=False, domain=bool, description='Print timing after writing each section of the LP file'))
    CONFIG.declare('skip_trivial_constraints', ConfigValue(default=False, domain=bool, description='Skip writing constraints whose body is constant'))
    CONFIG.declare('file_determinism', ConfigValue(default=FileDeterminism.ORDERED, domain=InEnum(FileDeterminism), description='How much effort to ensure file is deterministic', doc='\n            How much effort do we want to put into ensuring the\n            LP file is written deterministically for a Pyomo model:\n                NONE (0) : None\n                ORDERED (10): rely on underlying component ordering (default)\n                SORT_INDICES (20) : sort keys of indexed components\n                SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)\n            '))
    CONFIG.declare('symbolic_solver_labels', ConfigValue(default=False, domain=bool, description='Write variables/constraints using model names', doc='\n            Export variables and constraints to the LP file using human-readable\n            text names derived from the corresponding Pyomo component names.\n            '))
    CONFIG.declare('row_order', ConfigValue(default=None, description='Preferred constraint ordering', doc='\n            List of constraints in the order that they should appear in the\n            LP file.  Unspecified constraints will appear at the end.'))
    CONFIG.declare('column_order', ConfigValue(default=None, description='Preferred variable ordering', doc='\n\n\n            List of variables in the order that they should appear in\n            the LP file.  Note that this is only a suggestion, as the LP\n            file format is row-major and the columns are inferred from\n            the order in which variables appear in the objective\n            followed by each constraint.'))
    CONFIG.declare('labeler', ConfigValue(default=None, description='Callable to use to generate symbol names in LP file', doc='\n            Export variables and constraints to the LP file using human-readable\n            text names derived from the corresponding Pyomo component names.\n            '))
    CONFIG.declare('output_fixed_variable_bounds', ConfigValue(default=False, domain=bool, description='DEPRECATED option from LPv1 that has no effect in the LPv2'))
    CONFIG.declare('allow_quadratic_objective', ConfigValue(default=True, domain=bool, description='If True, allow quadratic terms in the model objective'))
    CONFIG.declare('allow_quadratic_constraint', ConfigValue(default=True, domain=bool, description='If True, allow quadratic terms in the model constraints'))

    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + '.lp'
        io_options = dict(io_options)
        qp = solver_capability('quadratic_objective')
        if 'allow_quadratic_objective' not in io_options:
            io_options['allow_quadratic_objective'] = qp
        qc = solver_capability('quadratic_constraint')
        if 'allow_quadratic_constraint' not in io_options:
            io_options['allow_quadratic_constraint'] = qc
        with open(filename, 'w', newline='') as FILE:
            info = self.write(model, FILE, **io_options)
        return (filename, info.symbol_map)

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream, **options):
        """Write a model in LP format.

        Returns
        -------
        LPWriterInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: io.TextIOBase
            The text output stream where the LP "file" will be written.
            Could be an opened file or a io.StringIO.

        """
        config = self.config(options)
        if config.output_fixed_variable_bounds:
            deprecation_warning("The 'output_fixed_variable_bounds' option to the LP writer is deprecated and is ignored by the lp_v2 writer.")
        with PauseGC():
            return _LPWriter_impl(ostream, config).write(model)