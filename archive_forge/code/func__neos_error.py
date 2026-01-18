import logging
import os
import re
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.opt import SolverFactory, SolverManagerFactory, OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import AsynchronousSolverManager
from pyomo.core.base import Block
import pyomo.neos.kestrel
def _neos_error(msg, results, current_message):
    error_re = re.compile('error', flags=re.I)
    warn_re = re.compile('warn', flags=re.I)
    logger.error('%s  NEOS log:\n%s' % (msg, current_message), exc_info=sys.exc_info())
    soln_data = results.data.decode('utf-8')
    for line in soln_data.splitlines():
        if error_re.search(line):
            logger.error(line)
        elif warn_re.search(line):
            logger.warning(line)