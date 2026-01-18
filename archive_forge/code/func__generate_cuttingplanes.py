from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _generate_cuttingplanes(self, instance_rBigM, cuts_obj, instance_rHull, var_info, transBlockName):
    opt = SolverFactory(self._config.solver)
    stream_solver = self._config.stream_solver
    opt.options = dict(self._config.solver_options)
    improving = True
    prev_obj = None
    epsilon = self._config.minimum_improvement_threshold
    cuts = None
    transBlock_rHull = instance_rHull.component(transBlockName)
    rBigM_obj, rBigM_linear_constraints = self._get_rBigM_obj_and_constraints(instance_rBigM)
    rHull_vars = [i for i in instance_rHull.component_data_objects(Var, descend_into=Block, sort=SortComponents.deterministic)]
    disaggregated_vars = self._get_disaggregated_vars(instance_rHull)
    hull_to_bigm_map = self._create_hull_to_bigm_substitution_map(var_info)
    bigm_to_hull_map = self._create_bigm_to_hull_substitution_map(var_info)
    xhat = ComponentMap()
    while improving:
        results = opt.solve(instance_rBigM, tee=stream_solver, load_solutions=False)
        if verify_successful_solve(results) is not NORMAL:
            logger.warning('Relaxed BigM subproblem did not solve normally. Stopping cutting plane generation.\n\n%s' % (results,))
            return
        instance_rBigM.solutions.load_from(results)
        rBigM_objVal = value(rBigM_obj)
        logger.warning('rBigM objective = %s' % (rBigM_objVal,))
        if transBlock_rHull.component('separation_objective') is None:
            self._add_separation_objective(var_info, transBlock_rHull)
        logger.info('x* is:')
        for x_rbigm, x_hull, x_star in var_info:
            if not x_rbigm.stale:
                x_star.set_value(x_rbigm.value)
                x_hull.set_value(x_rbigm.value, skip_validation=True)
            if self.verbose:
                logger.info('\t%s = %s' % (x_rbigm.getname(fully_qualified=True), x_rbigm.value))
        if prev_obj is None:
            improving = True
        else:
            obj_diff = prev_obj - rBigM_objVal
            improving = abs(obj_diff) > epsilon if abs(rBigM_objVal) < 1 else abs(obj_diff / prev_obj) > epsilon
        results = opt.solve(instance_rHull, tee=stream_solver, load_solutions=False)
        if verify_successful_solve(results) is not NORMAL:
            logger.warning('Hull separation subproblem did not solve normally. Stopping cutting plane generation.\n\n%s' % (results,))
            return
        instance_rHull.solutions.load_from(results)
        logger.warning('separation problem objective value: %s' % value(transBlock_rHull.separation_objective))
        if self.verbose:
            logger.info('xhat is: ')
        for x_rbigm, x_hull, x_star in var_info:
            xhat[x_rbigm] = value(x_hull)
            if self.verbose:
                logger.info('\t%s = %s' % (x_hull.getname(fully_qualified=True), x_hull.value))
        if value(transBlock_rHull.separation_objective) < self._config.separation_objective_threshold:
            logger.warning('Separation problem objective below threshold of %s: Stopping cut generation.' % self._config.separation_objective_threshold)
            break
        cuts = self._config.create_cuts(transBlock_rHull, var_info, hull_to_bigm_map, rBigM_linear_constraints, rHull_vars, disaggregated_vars, self._config.norm, self._config.cut_filtering_threshold, self._config.zero_tolerance, self._config.do_integer_arithmetic, self._config.tight_constraint_tolerance)
        if cuts is None:
            logger.warning('Did not generate a valid cut, stopping cut generation.')
            break
        if not improving:
            logger.warning('Difference in relaxed BigM problem objective values from past two iterations is below threshold of %s: Stopping cut generation.' % epsilon)
            break
        for cut in cuts:
            cut_number = len(cuts_obj)
            logger.warning('Adding cut %s to BigM model.' % (cut_number,))
            cuts_obj.add(cut_number, cut)
            if self._config.post_process_cut is not None:
                self._config.post_process_cut(cuts_obj[cut_number], transBlock_rHull, bigm_to_hull_map, opt, stream_solver, self._config.back_off_problem_tolerance)
        if cut_number + 1 == self._config.max_number_of_cuts:
            logger.warning('Reached maximum number of cuts.')
            break
        prev_obj = rBigM_objVal
        for x_rbigm, x_hull, x_star in var_info:
            x_rbigm.set_value(xhat[x_rbigm], skip_validation=True)