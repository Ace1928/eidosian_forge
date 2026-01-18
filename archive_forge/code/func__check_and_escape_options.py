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
def _check_and_escape_options(options):
    for key, val in self.options.items():
        tmp_k = str(key)
        _bad = ' ' in tmp_k
        tmp_v = str(val)
        if ' ' in tmp_v:
            if '"' in tmp_v:
                if "'" in tmp_v:
                    _bad = True
                else:
                    tmp_v = "'" + tmp_v + "'"
            else:
                tmp_v = '"' + tmp_v + '"'
        if _bad:
            raise ValueError('Unable to properly escape solver option:\n\t%s=%s' % (key, val))
        yield (tmp_k, tmp_v)