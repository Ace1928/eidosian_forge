from pyomo.common.dependencies import attempt_import
import itertools
import logging
from operator import attrgetter
from pyomo.common import DeveloperError
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.collections import ComponentMap
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.core.base import (
from pyomo.core.base.boolean_var import (
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import IndexedParam, ScalarParam, _ParamData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.core.base import Set, RangeSet
from pyomo.core.base.set import SetProduct
from pyomo.opt import WriterFactory, SolverFactory, TerminationCondition, SolverResults
from pyomo.network import Port
@SolverFactory.register('cp_optimizer', doc='Direct interface to CPLEX CP Optimizer')
class CPOptimizerSolver(object):
    CONFIG = ConfigDict('cp_optimizer_solver')
    CONFIG.declare('symbolic_solver_labels', ConfigValue(default=False, domain=bool, description='Write Pyomo Var and Constraint names to docplex model'))
    CONFIG.declare('tee', ConfigValue(default=False, domain=bool, description='Stream solver output to terminal.'))
    CONFIG.declare('options', ConfigValue(default={}, description='Dictionary of solver options.'))
    _unrestricted_license = None

    def __init__(self, **kwds):
        self.config = self.CONFIG()
        self.config.set_value(kwds)
        if docplex_available:
            self._solve_status_map = {cp.SOLVE_STATUS_UNKNOWN: TerminationCondition.unknown, cp.SOLVE_STATUS_INFEASIBLE: TerminationCondition.infeasible, cp.SOLVE_STATUS_FEASIBLE: TerminationCondition.feasible, cp.SOLVE_STATUS_OPTIMAL: TerminationCondition.optimal, cp.SOLVE_STATUS_JOB_ABORTED: None, cp.SOLVE_STATUS_JOB_FAILED: TerminationCondition.solverFailure}
            self._stop_cause_map = {cp.STOP_CAUSE_NOT_STOPPED: TerminationCondition.unknown, cp.STOP_CAUSE_LIMIT: TerminationCondition.maxTimeLimit, cp.STOP_CAUSE_EXIT: TerminationCondition.userInterrupt, cp.STOP_CAUSE_ABORT: TerminationCondition.userInterrupt}

    @property
    def options(self):
        return self.config.options

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        return Executable('cpoptimizer').available() and docplex_available

    def license_is_valid(self):
        if CPOptimizerSolver._unrestricted_license is None:
            x = cp.integer_var_list(141, 1, 141, 'X')
            m = cp.CpoModel()
            m.add(cp.all_diff(x))
            try:
                m.solve()
                CPOptimizerSolver._unrestricted_license = True
            except cp_solver.solver.CpoSolverException:
                CPOptimizerSolver._unrestricted_license = False
        return CPOptimizerSolver._unrestricted_license

    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.config()
        config.set_value(kwds)
        writer = DocplexWriter()
        cpx_model, var_map = writer.write(model, symbolic_solver_labels=config.symbolic_solver_labels)
        if not config.tee:
            verbosity = config.options.get('LogVerbosity')
            if verbosity is None:
                config.options['LogVerbosity'] = 'Quiet'
        msol = cpx_model.solve(**self.options)
        results = SolverResults()
        results.solver.name = 'CP Optimizer'
        results.problem.name = model.name
        info = msol.get_solver_infos()
        results.problem.number_of_constraints = info.get_number_of_constraints()
        int_vars = info.get_number_of_integer_vars()
        interval_vars = info.get_number_of_interval_vars()
        results.problem.number_of_integer_vars = int_vars
        results.problem.number_of_interval_vars = interval_vars
        results.problem.number_of_variables = int_vars + interval_vars
        val = msol.get_objective_value()
        bound = msol.get_objective_bound()
        if cpx_model.is_maximization():
            results.problem.number_of_objectives = 1
            results.problem.sense = maximize
            results.problem.lower_bound = val
            results.problem.upper_bound = bound
        elif cpx_model.is_minimization():
            results.problem.number_of_objectives = 1
            results.problem.sense = minimize
            results.problem.lower_bound = bound
            results.problem.upper_bound = val
        else:
            results.problem.number_of_objectives = 0
            results.problem.sense = None
            results.problem.lower_bound = None
            results.problem.upper_bound = None
        results.solver.solve_time = msol.get_solve_time()
        solve_status = msol.get_solve_status()
        results.solver.termination_condition = self._solve_status_map[solve_status] if solve_status is not None else self._stop_cause_map[msol.get_stop_cause()]
        cp_sol = msol.get_solution()
        if cp_sol is not None:
            for py_var, cp_var in var_map.items():
                sol = cp_sol.get_var_solution(cp_var)
                if sol is None:
                    logger.warning("CP optimizer did not return a value for variable '%s'" % py_var.name)
                else:
                    sol = sol.get_value()
                if py_var.ctype is IntervalVar:
                    if len(sol) == 0:
                        py_var.is_present.set_value(False)
                    else:
                        start, end, size = sol
                        py_var.is_present.set_value(True)
                        py_var.start_time.set_value(start, skip_validation=True)
                        py_var.end_time.set_value(end, skip_validation=True)
                        py_var.length.set_value(end - start, skip_validation=True)
                elif py_var.ctype in {Var, BooleanVar}:
                    py_var.set_value(sol, skip_validation=True)
                else:
                    raise DeveloperError('Unrecognized Pyomo type in pyomo-to-docplex variable map: %s' % type(py_var))
        return results