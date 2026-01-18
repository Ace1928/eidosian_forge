import os
import subprocess
import datetime
import io
from typing import Mapping, Optional, Sequence
from pyomo.common import Executable
from pyomo.common.config import ConfigValue, document_kwargs_from_configdict, ConfigDict
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.config import SolverConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.contrib.solver.results import Results, TerminationCondition, SolutionStatus
from pyomo.contrib.solver.sol_reader import parse_sol_file
from pyomo.contrib.solver.solution import SolSolutionLoader
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap
import logging
def _create_command_line(self, basename: str, config: IpoptConfig, opt_file: bool):
    cmd = [str(config.executable), basename + '.nl', '-AMPL']
    if opt_file:
        cmd.append('option_file_name=' + basename + '.opt')
    if 'option_file_name' in config.solver_options:
        raise ValueError('Pyomo generates the ipopt options file as part of the `solve` method. Add all options to ipopt.config.solver_options instead.')
    if config.time_limit is not None and 'max_cpu_time' not in config.solver_options:
        config.solver_options['max_cpu_time'] = config.time_limit
    for k, val in config.solver_options.items():
        if k in ipopt_command_line_options:
            cmd.append(str(k) + '=' + str(val))
    return cmd