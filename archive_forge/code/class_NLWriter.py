import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
@WriterFactory.register('nl_v2', 'Generate the corresponding AMPL NL file (version 2).')
class NLWriter(object):
    CONFIG = ConfigBlock('nlwriter')
    CONFIG.declare('show_section_timing', ConfigValue(default=False, domain=bool, description='Print timing after writing each section of the NL file'))
    CONFIG.declare('skip_trivial_constraints', ConfigValue(default=True, domain=bool, description='Skip writing constraints whose body is constant'))
    CONFIG.declare('file_determinism', ConfigValue(default=FileDeterminism.ORDERED, domain=InEnum(FileDeterminism), description='How much effort to ensure file is deterministic', doc='\n        How much effort do we want to put into ensuring the\n        NL file is written deterministically for a Pyomo model:\n            NONE (0) : None\n            ORDERED (10): rely on underlying component ordering (default)\n            SORT_INDICES (20) : sort keys of indexed components\n            SORT_SYMBOLS (30) : sort keys AND sort names (not declaration order)\n        '))
    CONFIG.declare('symbolic_solver_labels', ConfigValue(default=False, domain=bool, description='Write the corresponding .row and .col files'))
    CONFIG.declare('scale_model', ConfigValue(default=True, domain=bool, description='Write variables and constraints in scaled space', doc="\n            If True, then the writer will output the model constraints and\n            variables in 'scaled space' using the scaling from the\n            'scaling_factor' Suffix, if provided."))
    CONFIG.declare('export_nonlinear_variables', ConfigValue(default=None, domain=list, description='Extra variables to include in NL file', doc="\n        List of variables to ensure are in the NL file (even if they\n        don't appear in any constraints)."))
    CONFIG.declare('row_order', ConfigValue(default=None, description='Preferred constraint ordering', doc='\n        List of constraints in the order that they should appear in the\n        NL file.  Note that this is only a suggestion, as the NL writer\n        will move all nonlinear constraints before linear ones\n        (preserving row_order within each group).'))
    CONFIG.declare('column_order', ConfigValue(default=None, description='Preferred variable ordering', doc='\n        List of variables in the order that they should appear in the NL\n        file.  Note that this is only a suggestion, as the NL writer\n        will move all nonlinear variables before linear ones, and within\n        nonlinear variables, variables appearing in both objectives and\n        constraints before variables appearing only in constraints,\n        which appear before variables appearing only in objectives.\n        Within each group, continuous variables appear before discrete\n        variables.  In all cases, column_order is preserved within each\n        group.'))
    CONFIG.declare('export_defined_variables', ConfigValue(default=True, domain=bool, description='Preferred variable ordering', doc="\n        If True, export Expression objects to the NL file as 'defined\n        variables'."))
    CONFIG.declare('linear_presolve', ConfigValue(default=True, domain=bool, description='Perform linear presolve', doc='\n        If True, we will perform a basic linear presolve by performing\n        variable elimination (without fill-in).'))

    def __init__(self):
        self.config = self.CONFIG()

    def __call__(self, model, filename, solver_capability, io_options):
        if filename is None:
            filename = model.name + '.nl'
        filename_base = os.path.splitext(filename)[0]
        row_fname = filename_base + '.row'
        col_fname = filename_base + '.col'
        config = self.config(io_options)
        config.scale_model = False
        config.linear_presolve = False
        config.skip_trivial_constraints = False
        if config.symbolic_solver_labels:
            _open = lambda fname: open(fname, 'w')
        else:
            _open = nullcontext
        with open(filename, 'w', newline='') as FILE, _open(row_fname) as ROWFILE, _open(col_fname) as COLFILE:
            info = self.write(model, FILE, ROWFILE, COLFILE, config=config)
        if not info.variables:
            os.remove(filename)
            if config.symbolic_solver_labels:
                os.remove(row_fname)
                os.remove(col_fname)
            raise ValueError('No variables appear in the Pyomo model constraints or objective. This is not supported by the NL file interface')
        set_pyomo_amplfunc_env(info.external_function_libraries)
        symbol_map = self._generate_symbol_map(info)
        return (filename, symbol_map)

    @document_kwargs_from_configdict(CONFIG)
    def write(self, model, ostream, rowstream=None, colstream=None, **options) -> NLWriterInfo:
        """Write a model in NL format.

        Returns
        -------
        NLWriterInfo

        Parameters
        ----------
        model: ConcreteModel
            The concrete Pyomo model to write out.

        ostream: io.TextIOBase
            The text output stream where the NL "file" will be written.
            Could be an opened file or a io.StringIO.

        rowstream: io.TextIOBase
            A text output stream to write the ASL "row file" (list of
            constraint / objective names).  Ignored unless
            `symbolic_solver_labels` is True.

        colstream: io.TextIOBase
            A text output stream to write the ASL "col file" (list of
            variable names).  Ignored unless `symbolic_solver_labels` is True.

        """
        config = options.pop('config', self.config)(options)
        with _NLWriter_impl(ostream, rowstream, colstream, config) as impl:
            return impl.write(model)

    def _generate_symbol_map(self, info):
        symbol_map = SymbolMap()
        symbol_map.addSymbols(((info, f'v{idx}') for idx, info in enumerate(info.variables)))
        symbol_map.addSymbols(((info, f'c{idx}') for idx, info in enumerate(info.constraints)))
        symbol_map.addSymbols(((info, f'o{idx}') for idx, info in enumerate(info.objectives)))
        return symbol_map