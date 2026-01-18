import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
@SolverFactory.register('_mock_cbc')
class MockCBC(CBCSHELL, MockMIP):
    """A Mock CBC solver used for testing"""

    def __init__(self, **kwds):
        try:
            CBCSHELL.__init__(self, **kwds)
        except ApplicationError:
            pass
        MockMIP.__init__(self, 'cbc')

    def available(self, exception_flag=True):
        return CBCSHELL.available(self, exception_flag)

    def create_command_line(self, executable, problem_files):
        command = CBCSHELL.create_command_line(self, executable, problem_files)
        MockMIP.create_command_line(self, executable, problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self, cmd):
        return MockMIP._execute_command(self, cmd)

    def _convert_problem(self, args, pformat, valid_pformats):
        if pformat in [ProblemFormat.mps, ProblemFormat.cpxlp, ProblemFormat.nl]:
            return (args, pformat, None)
        else:
            return (args, ProblemFormat.mps, None)

    def _get_version(self):
        return (2, 9, 9)