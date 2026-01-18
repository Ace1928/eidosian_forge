import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class TestIterationLogRecord(unittest.TestCase):
    """
    Test the PyROS `IterationLogRecord` class.
    """

    def test_log_header(self):
        """Test method for logging iteration log table header."""
        ans = '------------------------------------------------------------------------------\nItn  Objective    1-Stg Shift  2-Stg Shift  #CViol  Max Viol     Wall Time (s)\n------------------------------------------------------------------------------\n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            IterationLogRecord.log_header(logger.info)
        self.assertEqual(LOG.getvalue(), ans, msg='Messages logged for iteration table header do not match expected result')

    def test_log_standard_iter_record(self):
        """Test logging function for PyROS IterationLogRecord."""
        iter_record = IterationLogRecord(iteration=4, objective=1.234567, first_stage_var_shift=2.3456789e-08, second_stage_var_shift=3.456789e-07, dr_var_shift=1.234567e-07, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=True, all_sep_problems_solved=True, global_separation=False)
        ans = '4     1.2346e+00  2.3457e-08   3.4568e-07   10      7.6543e-03   21.200       \n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()
        self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')

    def test_log_iter_record_polishing_failed(self):
        """Test iteration log record in event of polishing failure."""
        iter_record = IterationLogRecord(iteration=4, objective=1.234567, first_stage_var_shift=2.3456789e-08, second_stage_var_shift=3.456789e-07, dr_var_shift=1.234567e-07, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=False, all_sep_problems_solved=True, global_separation=False)
        ans = '4     1.2346e+00  2.3457e-08   3.4568e-07*  10      7.6543e-03   21.200       \n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()
        self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')

    def test_log_iter_record_global_separation(self):
        """
        Test iteration log record in event global separation performed.
        In this case, a 'g' should be appended to the max violation
        reported. Useful in the event neither local nor global separation
        was bypassed.
        """
        iter_record = IterationLogRecord(iteration=4, objective=1.234567, first_stage_var_shift=2.3456789e-08, second_stage_var_shift=3.456789e-07, dr_var_shift=1.234567e-07, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=True, all_sep_problems_solved=True, global_separation=True)
        ans = '4     1.2346e+00  2.3457e-08   3.4568e-07   10      7.6543e-03g  21.200       \n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()
        self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')

    def test_log_iter_record_not_all_sep_solved(self):
        """
        Test iteration log record in event not all separation problems
        were solved successfully. This may have occurred if the PyROS
        solver time limit was reached, or the user-provides subordinate
        optimizer(s) were unable to solve a separation subproblem
        to an acceptable level.
        A '+' should be appended to the number of performance constraints
        found to be violated.
        """
        iter_record = IterationLogRecord(iteration=4, objective=1.234567, first_stage_var_shift=2.3456789e-08, second_stage_var_shift=3.456789e-07, dr_var_shift=1.234567e-07, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=True, all_sep_problems_solved=False, global_separation=False)
        ans = '4     1.2346e+00  2.3457e-08   3.4568e-07   10+     7.6543e-03   21.200       \n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()
        self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')

    def test_log_iter_record_all_special(self):
        """
        Test iteration log record in event DR polishing and global
        separation failed.
        """
        iter_record = IterationLogRecord(iteration=4, objective=1.234567, first_stage_var_shift=2.3456789e-08, second_stage_var_shift=3.456789e-07, dr_var_shift=1.234567e-07, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=False, all_sep_problems_solved=False, global_separation=True)
        ans = '4     1.2346e+00  2.3457e-08   3.4568e-07*  10+     7.6543e-03g  21.200       \n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()
        self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')

    def test_log_iter_record_attrs_none(self):
        """
        Test logging of iteration record in event some
        attributes are of value `None`. In this case, a '-'
        should be printed in lieu of a numerical value.
        Example where this occurs: the first iteration,
        in which there is no first-stage shift or DR shift.
        """
        iter_record = IterationLogRecord(iteration=0, objective=-1.234567, first_stage_var_shift=None, second_stage_var_shift=None, dr_var_shift=None, num_violated_cons=10, max_violation=0.007654321, elapsed_time=21.2, dr_polishing_success=True, all_sep_problems_solved=False, global_separation=True)
        ans = '0    -1.2346e+00  -            -            10+     7.6543e-03g  21.200       \n'
        with LoggingIntercept(level=logging.INFO) as LOG:
            iter_record.log(logger.info)
        result = LOG.getvalue()
        self.assertEqual(ans, result, msg='Iteration log record message does not match expected result')