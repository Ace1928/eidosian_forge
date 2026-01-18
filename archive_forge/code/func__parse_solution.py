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
def _parse_solution(self, instream: io.TextIOBase, nl_info: NLWriterInfo):
    results = Results()
    res, sol_data = parse_sol_file(sol_file=instream, nl_info=nl_info, result=results)
    if res.solution_status == SolutionStatus.noSolution:
        res.solution_loader = SolSolutionLoader(None, None)
    else:
        res.solution_loader = IpoptSolutionLoader(sol_data=sol_data, nl_info=nl_info)
    return res