import logging
import os
import re
import sys
import time
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var
def _get_nlp_results(self, results, soln):
    """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
    xprob = self._solver_model
    xp = xpress
    xprob_attrs = xprob.attributes
    solver = xprob_attrs.xslp_solverselected
    if solver == 2:
        if xprob_attrs.originalmipents > 0 or xprob_attrs.originalsets > 0:
            return self._get_mip_results(results, soln)
        elif xprob_attrs.lpstatus and (not xprob_attrs.xslp_nlpstatus):
            return self._get_lp_results(results, soln)
    status = xprob_attrs.xslp_nlpstatus
    solstatus = xprob_attrs.xslp_solstatus
    have_soln = False
    optimal = False
    if status == xp.nlp_unstarted:
        results.solver.status = SolverStatus.unknown
        results.solver.termination_message = 'Non-convex model solve was not started'
        results.solver.termination_condition = TerminationCondition.unknown
        soln.status = SolutionStatus.unknown
    elif status == xp.nlp_locally_optimal:
        if solstatus in [2, 3]:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = 'Non-convex model was solved to local optimality'
            results.solver.termination_condition = TerminationCondition.locallyOptimal
            soln.status = SolutionStatus.locallyOptimal
        else:
            results.solver.status = SolverStatus.ok
            results.solver.termination_message = 'Feasible solution found for non-convex model'
            results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible
        have_soln = True
    elif status == xp.nlp_globally_optimal:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Non-convex model was solved to global optimality'
        results.solver.termination_condition = TerminationCondition.optimal
        soln.status = SolutionStatus.optimal
        have_soln = True
        optimal = True
    elif status == xp.nlp_locally_infeasible:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Non-convex model was proven to be locally infeasible'
        results.solver.termination_condition = TerminationCondition.noSolution
        soln.status = SolutionStatus.unknown
    elif status == xp.nlp_infeasible:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Non-convex model was proven to be infeasible'
        results.solver.termination_condition = TerminationCondition.infeasible
        soln.status = SolutionStatus.infeasible
    elif status == xp.nlp_unbounded:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Non-convex model is locally unbounded'
        results.solver.termination_condition = TerminationCondition.unbounded
        soln.status = SolutionStatus.unbounded
    elif status == xp.nlp_unfinished:
        results.solver.status = SolverStatus.ok
        results.solver.termination_message = 'Non-convex solve not finished (numerical issues?)'
        results.solver.termination_condition = TerminationCondition.unknown
        soln.status = SolutionStatus.unknown
        have_soln = True
    else:
        results.solver.status = SolverStatus.error
        results.solver.termination_message = 'Error for non-convex model: ' + str(status)
        results.solver.termination_condition = TerminationCondition.error
        soln.status = SolutionStatus.error
    results.problem.upper_bound = None
    results.problem.lower_bound = None
    try:
        if xprob_attrs.objsense > 0.0 or optimal:
            results.problem.upper_bound = xprob_attrs.xslp_objval
        if xprob_attrs.objsense < 0.0 or optimal:
            results.problem.lower_bound = xprob_attrs.xslp_objval
    except (XpressDirect.XpressException, AttributeError):
        pass
    return have_soln