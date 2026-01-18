from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def _create_doe_model(self, no_obj=True):
    """
        Add equations to compute sensitivities, FIM, and objective.

        Parameters
        -----------
        no_obj: if True, objective function is 0.

        Return
        -------
        model: the DOE model
        """
    model = self._create_block()
    model.regression_parameters = pyo.Set(initialize=list(self.param.keys()))
    model.measured_variables = pyo.Set(initialize=self.measure_name)

    def identity_matrix(m, i, j):
        if i == j:
            return 1
        else:
            return 0
    model.sensitivity_jacobian = pyo.Var(model.regression_parameters, model.measured_variables, initialize=0.1)
    if self.fim_initial:
        dict_fim_initialize = {}
        for i, bu in enumerate(model.regression_parameters):
            for j, un in enumerate(model.regression_parameters):
                dict_fim_initialize[bu, un] = self.fim_initial[i][j]

    def initialize_fim(m, j, d):
        return dict_fim_initialize[j, d]
    if self.fim_initial:
        model.fim = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=initialize_fim)
    else:
        model.fim = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=identity_matrix)
    if type(self.L_initial) != type(None):
        dict_cho = {}
        for i, bu in enumerate(model.regression_parameters):
            for j, un in enumerate(model.regression_parameters):
                dict_cho[bu, un] = self.L_initial[i][j]

    def init_cho(m, i, j):
        return dict_cho[i, j]
    if self.Cholesky_option:
        if type(self.L_initial) != type(None):
            model.L_ele = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=init_cho)
        else:
            model.L_ele = pyo.Var(model.regression_parameters, model.regression_parameters, initialize=identity_matrix)
        for i, c in enumerate(model.regression_parameters):
            for j, d in enumerate(model.regression_parameters):
                if i < j:
                    model.L_ele[c, d].fix(0.0)
                if self.L_LB:
                    if c == d:
                        model.L_ele[c, d].setlb(self.L_LB)

    def jacobian_rule(m, p, n):
        """
            m: Pyomo model
            p: parameter
            n: response
            """
        cuid = pyo.ComponentUID(n)
        var_up = cuid.find_component_on(m.block[self.scenario_num[p][0]])
        var_lo = cuid.find_component_on(m.block[self.scenario_num[p][1]])
        if self.scale_nominal_param_value:
            return m.sensitivity_jacobian[p, n] == (var_up - var_lo) / self.eps_abs[p] * self.param[p] * self.scale_constant_value
        else:
            return m.sensitivity_jacobian[p, n] == (var_up - var_lo) / self.eps_abs[p] * self.scale_constant_value
    fim_initial_dict = {}
    for i, bu in enumerate(model.regression_parameters):
        for j, un in enumerate(model.regression_parameters):
            fim_initial_dict[bu, un] = self.prior_FIM[i][j]

    def read_prior(m, i, j):
        return fim_initial_dict[i, j]
    model.priorFIM = pyo.Expression(model.regression_parameters, model.regression_parameters, rule=read_prior)

    def fim_rule(m, p, q):
        """
            m: Pyomo model
            p: parameter
            q: parameter
            """
        return m.fim[p, q] == sum((1 / self.measurement_vars.variance[n] * m.sensitivity_jacobian[p, n] * m.sensitivity_jacobian[q, n] for n in model.measured_variables)) + m.priorFIM[p, q] * self.fim_scale_constant_value
    model.jacobian_constraint = pyo.Constraint(model.regression_parameters, model.measured_variables, rule=jacobian_rule)
    model.fim_constraint = pyo.Constraint(model.regression_parameters, model.regression_parameters, rule=fim_rule)
    return model