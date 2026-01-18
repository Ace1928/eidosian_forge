import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def _Q_at_theta(self, thetavals, initialize_parmest_model=False):
    """
        Return the objective function value with fixed theta values.

        Parameters
        ----------
        thetavals: dict
            A dictionary of theta values.

        initialize_parmest_model: boolean
            If True: Solve square problem instance, build extensive form of the model for
            parameter estimation, and set flag model_initialized to True

        Returns
        -------
        objectiveval: float
            The objective function value.
        thetavals: dict
            A dictionary of all values for theta that were input.
        solvertermination: Pyomo TerminationCondition
            Tries to return the "worst" solver status across the scenarios.
            pyo.TerminationCondition.optimal is the best and
            pyo.TerminationCondition.infeasible is the worst.
        """
    optimizer = pyo.SolverFactory('ipopt')
    if len(thetavals) > 0:
        dummy_cb = {'callback': self._instance_creation_callback, 'ThetaVals': thetavals, 'theta_names': self._return_theta_names(), 'cb_data': self.callback_data}
    else:
        dummy_cb = {'callback': self._instance_creation_callback, 'theta_names': self._return_theta_names(), 'cb_data': self.callback_data}
    if self.diagnostic_mode:
        if len(thetavals) > 0:
            print('    Compute objective at theta = ', str(thetavals))
        else:
            print('    Compute objective at initial theta')
    instance = _experiment_instance_creation_callback('FOO0', None, dummy_cb)
    try:
        first = next(instance.component_objects(pyo.Constraint, active=True))
        active_constraints = True
    except:
        active_constraints = False
    WorstStatus = pyo.TerminationCondition.optimal
    totobj = 0
    scenario_numbers = list(range(len(self.callback_data)))
    if initialize_parmest_model:
        scen_dict = dict()
    for snum in scenario_numbers:
        sname = 'scenario_NODE' + str(snum)
        instance = _experiment_instance_creation_callback(sname, None, dummy_cb)
        if initialize_parmest_model:
            theta_init_vals = []
            theta_ref = self._return_theta_names()
            for i, theta in enumerate(theta_ref):
                var_cuid = ComponentUID(theta)
                var_validate = var_cuid.find_component_on(instance)
                if var_validate is None:
                    logger.warning('theta_name %s was not found on the model', theta)
                else:
                    try:
                        if len(thetavals) == 0:
                            var_validate.fix()
                        else:
                            var_validate.fix(thetavals[theta])
                        theta_init_vals.append(var_validate)
                    except:
                        logger.warning('Unable to fix model parameter value for %s (not a Pyomo model Var)', theta)
        if active_constraints:
            if self.diagnostic_mode:
                print('      Experiment = ', snum)
                print('     First solve with special diagnostics wrapper')
                status_obj, solved, iters, time, regu = utils.ipopt_solve_with_stats(instance, optimizer, max_iter=500, max_cpu_time=120)
                print('   status_obj, solved, iters, time, regularization_stat = ', str(status_obj), str(solved), str(iters), str(time), str(regu))
            results = optimizer.solve(instance)
            if self.diagnostic_mode:
                print('standard solve solver termination condition=', str(results.solver.termination_condition))
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                if WorstStatus != pyo.TerminationCondition.infeasible:
                    WorstStatus = results.solver.termination_condition
                if initialize_parmest_model:
                    if self.diagnostic_mode:
                        print('Scenario {:d} infeasible with initialized parameter values'.format(snum))
            elif initialize_parmest_model:
                if self.diagnostic_mode:
                    print('Scenario {:d} initialization successful with initial parameter values'.format(snum))
            if initialize_parmest_model:
                for theta in theta_init_vals:
                    theta.unfix()
                scen_dict[sname] = instance
        elif initialize_parmest_model:
            for theta in theta_init_vals:
                theta.unfix()
            scen_dict[sname] = instance
        objobject = getattr(instance, self._second_stage_cost_exp)
        objval = pyo.value(objobject)
        totobj += objval
    retval = totobj / len(scenario_numbers)
    if initialize_parmest_model and (not hasattr(self, 'ef_instance')):
        if len(scen_dict) > 0:
            for scen in scen_dict.values():
                scen._mpisppy_probability = 1 / len(scen_dict)
        if use_mpisppy:
            EF_instance = sputils._create_EF_from_scen_dict(scen_dict, EF_name='_Q_at_theta')
        else:
            EF_instance = local_ef._create_EF_from_scen_dict(scen_dict, EF_name='_Q_at_theta', nonant_for_fixed_vars=True)
        self.ef_instance = EF_instance
        self.model_initialized = True
        if len(thetavals) == 0:
            theta_ref = self._return_theta_names()
            for i, theta in enumerate(theta_ref):
                thetavals[theta] = theta_init_vals[i]()
    return (retval, thetavals, WorstStatus)