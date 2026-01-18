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
class IterationLogRecord:
    """
    PyROS solver iteration log record.

    Parameters
    ----------
    iteration : int or None, optional
        Iteration number.
    objective : int or None, optional
        Master problem objective value.
        Note: if the sense of the original model is maximization,
        then this is the negative of the objective value
        of the original model.
    first_stage_var_shift : float or None, optional
        Infinity norm of the difference between first-stage
        variable vectors for the current and previous iterations.
    second_stage_var_shift : float or None, optional
        Infinity norm of the difference between decision rule
        variable vectors for the current and previous iterations.
    dr_polishing_success : bool or None, optional
        True if DR polishing solved successfully, False otherwise.
    num_violated_cons : int or None, optional
        Number of performance constraints found to be violated
        during separation step.
    all_sep_problems_solved : int or None, optional
        True if all separation problems were solved successfully,
        False otherwise (such as if there was a time out, subsolver
        error, or only a subset of the problems were solved due to
        custom constraint prioritization).
    global_separation : bool, optional
        True if separation problems were solved with the subordinate
        global optimizer(s), False otherwise.
    max_violation : int or None
        Maximum scaled violation of any performance constraint
        found during separation step.
    elapsed_time : float, optional
        Total time elapsed up to the current iteration, in seconds.

    Attributes
    ----------
    iteration : int or None
        Iteration number.
    objective : int or None
        Master problem objective value.
        Note: if the sense of the original model is maximization,
        then this is the negative of the objective value
        of the original model.
    first_stage_var_shift : float or None
        Infinity norm of the relative difference between first-stage
        variable vectors for the current and previous iterations.
    second_stage_var_shift : float or None
        Infinity norm of the relative difference between second-stage
        variable vectors (evaluated subject to the nominal uncertain
        parameter realization) for the current and previous iterations.
    dr_var_shift : float or None
        Infinity norm of the relative difference between decision rule
        variable vectors for the current and previous iterations.
        NOTE: This value is not reported in log messages.
    dr_polishing_success : bool or None
        True if DR polishing was solved successfully, False otherwise.
    num_violated_cons : int or None
        Number of performance constraints found to be violated
        during separation step.
    all_sep_problems_solved : int or None
        True if all separation problems were solved successfully,
        False otherwise (such as if there was a time out, subsolver
        error, or only a subset of the problems were solved due to
        custom constraint prioritization).
    global_separation : bool
        True if separation problems were solved with the subordinate
        global optimizer(s), False otherwise.
    max_violation : int or None
        Maximum scaled violation of any performance constraint
        found during separation step.
    elapsed_time : float
        Total time elapsed up to the current iteration, in seconds.
    """
    _LINE_LENGTH = 78
    _ATTR_FORMAT_LENGTHS = {'iteration': 5, 'objective': 13, 'first_stage_var_shift': 13, 'second_stage_var_shift': 13, 'dr_var_shift': 13, 'num_violated_cons': 8, 'max_violation': 13, 'elapsed_time': 13}
    _ATTR_HEADER_NAMES = {'iteration': 'Itn', 'objective': 'Objective', 'first_stage_var_shift': '1-Stg Shift', 'second_stage_var_shift': '2-Stg Shift', 'dr_var_shift': 'DR Shift', 'num_violated_cons': '#CViol', 'max_violation': 'Max Viol', 'elapsed_time': 'Wall Time (s)'}

    def __init__(self, iteration, objective, first_stage_var_shift, second_stage_var_shift, dr_var_shift, dr_polishing_success, num_violated_cons, all_sep_problems_solved, global_separation, max_violation, elapsed_time):
        """Initialize self (see class docstring)."""
        self.iteration = iteration
        self.objective = objective
        self.first_stage_var_shift = first_stage_var_shift
        self.second_stage_var_shift = second_stage_var_shift
        self.dr_var_shift = dr_var_shift
        self.dr_polishing_success = dr_polishing_success
        self.num_violated_cons = num_violated_cons
        self.all_sep_problems_solved = all_sep_problems_solved
        self.global_separation = global_separation
        self.max_violation = max_violation
        self.elapsed_time = elapsed_time

    def get_log_str(self):
        """Get iteration log string."""
        attrs = ['iteration', 'objective', 'first_stage_var_shift', 'second_stage_var_shift', 'num_violated_cons', 'max_violation', 'elapsed_time']
        return ''.join((self._format_record_attr(attr) for attr in attrs))

    def _format_record_attr(self, attr_name):
        """Format attribute record for logging."""
        attr_val = getattr(self, attr_name)
        if attr_val is None:
            fmt_str = f'<{self._ATTR_FORMAT_LENGTHS[attr_name]}s'
            return f'{'-':{fmt_str}}'
        else:
            attr_val_fstrs = {'iteration': "f'{attr_val:d}'", 'objective': "f'{attr_val: .4e}'", 'first_stage_var_shift': "f'{attr_val:.4e}'", 'second_stage_var_shift': "f'{attr_val:.4e}'", 'dr_var_shift': "f'{attr_val:.4e}'", 'num_violated_cons': "f'{attr_val:d}'", 'max_violation': "f'{attr_val:.4e}'", 'elapsed_time': "f'{attr_val:.3f}'"}
            if attr_name in ['second_stage_var_shift', 'dr_var_shift']:
                qual = '*' if not self.dr_polishing_success else ''
            elif attr_name == 'num_violated_cons':
                qual = '+' if not self.all_sep_problems_solved else ''
            elif attr_name == 'max_violation':
                qual = 'g' if self.global_separation else ''
            else:
                qual = ''
            attr_val_str = f'{eval(attr_val_fstrs[attr_name])}{qual}'
            return f'{attr_val_str:{f'<{self._ATTR_FORMAT_LENGTHS[attr_name]}'}}'

    def log(self, log_func, **log_func_kwargs):
        """Log self."""
        log_str = self.get_log_str()
        log_func(log_str, **log_func_kwargs)

    @staticmethod
    def get_log_header_str():
        """Get string for iteration log header."""
        fmt_lengths_dict = IterationLogRecord._ATTR_FORMAT_LENGTHS
        header_names_dict = IterationLogRecord._ATTR_HEADER_NAMES
        return ''.join((f'{header_names_dict[attr]:<{fmt_lengths_dict[attr]}s}' for attr in fmt_lengths_dict if attr != 'dr_var_shift'))

    @staticmethod
    def log_header(log_func, with_rules=True, **log_func_kwargs):
        """Log header."""
        if with_rules:
            IterationLogRecord.log_header_rule(log_func, **log_func_kwargs)
        log_func(IterationLogRecord.get_log_header_str(), **log_func_kwargs)
        if with_rules:
            IterationLogRecord.log_header_rule(log_func, **log_func_kwargs)

    @staticmethod
    def log_header_rule(log_func, fillchar='-', **log_func_kwargs):
        """Log header rule."""
        log_func(fillchar * IterationLogRecord._LINE_LENGTH, **log_func_kwargs)