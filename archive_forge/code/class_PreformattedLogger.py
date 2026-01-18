import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
class PreformattedLogger(logging.Logger):
    """
    A specialized logger object designed to cast log messages
    to Pyomo `Preformatted` objects prior to logging the messages.
    Useful for circumventing the formatters of the standard Pyomo
    logger in the event an instance is a descendant of the Pyomo
    logger.
    """

    def critical(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.CRITICAL`.
        """
        return super(PreformattedLogger, self).critical(Preformatted(msg % args if args else msg), **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.ERROR`.
        """
        return super(PreformattedLogger, self).error(Preformatted(msg % args if args else msg), **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.WARNING`.
        """
        return super(PreformattedLogger, self).warning(Preformatted(msg % args if args else msg), **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.INFO`.
        """
        return super(PreformattedLogger, self).info(Preformatted(msg % args if args else msg), **kwargs)

    def debug(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.DEBUG`.
        """
        return super(PreformattedLogger, self).debug(Preformatted(msg % args if args else msg), **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with integer
        severity `level`.
        """
        return super(PreformattedLogger, self).log(level, Preformatted(msg % args if args else msg), **kwargs)