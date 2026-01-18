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
def _parse_ipopt_output(self, stream: io.StringIO):
    """
        Parse an IPOPT output file and return:

        * number of iterations
        * time in IPOPT

        """
    iters = None
    nofunc_time = None
    func_time = None
    total_time = None
    for line in stream.getvalue().splitlines():
        if line.startswith('Number of Iterations....:'):
            tokens = line.split()
            iters = int(tokens[-1])
        elif line.startswith('Total seconds in IPOPT                               ='):
            tokens = line.split()
            total_time = float(tokens[-1])
        elif line.startswith('Total CPU secs in IPOPT (w/o function evaluations)   ='):
            tokens = line.split()
            nofunc_time = float(tokens[-1])
        elif line.startswith('Total CPU secs in NLP function evaluations           ='):
            tokens = line.split()
            func_time = float(tokens[-1])
    return (iters, nofunc_time, func_time, total_time)