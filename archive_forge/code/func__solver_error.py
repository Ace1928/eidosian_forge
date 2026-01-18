import re
import sys
import time
import logging
import shlex
from pyomo.common import Factory
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat
import pyomo.opt.base.results
def _solver_error(self, method_name):
    raise RuntimeError('Attempting to use an unavailable solver.\n\nThe SolverFactory was unable to create the solver "%s"\nand returned an UnknownSolver object.  This error is raised at the point\nwhere the UnknownSolver object was used as if it were valid (by calling\nmethod "%s").\n\nThe original solver was created with the following parameters:\n\t' % (self.type, method_name) + '\n\t'.join(('%s: %s' % i for i in sorted(self._kwds.items()))) + '\n\t_args: %s' % (self._args,) + '\n\toptions: %s' % (self.options,))