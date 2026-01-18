import abc
import enum
from typing import Sequence, Dict, Optional, Mapping, NoReturn, List, Tuple
import os
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import ApplicationError
from pyomo.common.deprecation import deprecation_warning
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import Solution as LegacySolution
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.config import SolverConfig, PersistentSolverConfig
from pyomo.contrib.solver.util import get_objective
from pyomo.contrib.solver.results import (
def _map_config(self, tee, load_solutions, symbolic_solver_labels, timelimit, report_timing, raise_exception_on_nonoptimal_result, solver_io, suffixes, logfile, keepfiles, solnfile, options):
    """Map between legacy and new interface configuration options"""
    self.config = self.config()
    self.config.tee = tee
    self.config.load_solutions = load_solutions
    self.config.symbolic_solver_labels = symbolic_solver_labels
    self.config.time_limit = timelimit
    self.config.solver_options.set_value(options)
    self.config.raise_exception_on_nonoptimal_result = raise_exception_on_nonoptimal_result
    if solver_io is not None:
        raise NotImplementedError('Still working on this')
    if suffixes is not None:
        raise NotImplementedError('Still working on this')
    if logfile is not None:
        raise NotImplementedError('Still working on this')
    if keepfiles or 'keepfiles' in self.config:
        cwd = os.getcwd()
        deprecation_warning(f'`keepfiles` has been deprecated in the new solver interface. Use `working_dir` instead to designate a directory in which files should be generated and saved. Setting `working_dir` to `{cwd}`.', version='6.7.1')
        self.config.working_dir = cwd
    if solnfile is not None:
        if 'filename' in self.config:
            filename = os.path.splitext(solnfile)[0]
            self.config.filename = filename