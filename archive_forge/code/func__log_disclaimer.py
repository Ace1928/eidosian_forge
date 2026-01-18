import logging
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.collections import Bunch
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.contrib.pyros.util import time_code
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.config import pyros_config
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.core.base import Constraint
from datetime import datetime
def _log_disclaimer(self, logger, **log_kwargs):
    """
        Log PyROS solver disclaimer messages.

        Parameters
        ----------
        logger : logging.Logger
            Logger through which to emit messages.
        **log_kwargs : dict, optional
            Keyword arguments to ``logger.log()`` callable.
            Should not include `msg`.
        """
    disclaimer_header = ' DISCLAIMER '.center(self._LOG_LINE_LENGTH, '=')
    logger.log(msg=disclaimer_header, **log_kwargs)
    logger.log(msg='PyROS is still under development. ', **log_kwargs)
    logger.log(msg='Please provide feedback and/or report any issues by creating a ticket at', **log_kwargs)
    logger.log(msg='https://github.com/Pyomo/pyomo/issues/new/choose', **log_kwargs)
    logger.log(msg='=' * self._LOG_LINE_LENGTH, **log_kwargs)