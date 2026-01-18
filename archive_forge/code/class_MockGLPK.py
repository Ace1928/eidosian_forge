import logging
import re
import sys
import csv
import subprocess
from pyomo.common.tempfiles import TempfileManager
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.opt import (
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
@SolverFactory.register('_mock_glpk')
class MockGLPK(GLPKSHELL, MockMIP):
    """A Mock GLPK solver used for testing"""

    def __init__(self, **kwds):
        try:
            GLPKSHELL.__init__(self, **kwds)
        except ApplicationError:
            pass
        MockMIP.__init__(self, 'glpk')

    def available(self, exception_flag=True):
        return GLPKSHELL.available(self, exception_flag)

    def create_command_line(self, executable, problem_files):
        command = GLPKSHELL.create_command_line(self, executable, problem_files)
        MockMIP.create_command_line(self, executable, problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self, cmd):
        return MockMIP._execute_command(self, cmd)

    def _convert_problem(self, args, pformat, valid_pformats):
        if pformat in [ProblemFormat.mps, ProblemFormat.cpxlp]:
            return (args, pformat, None)
        else:
            return (args, ProblemFormat.cpxlp, None)