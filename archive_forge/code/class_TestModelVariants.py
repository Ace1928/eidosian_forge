from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestModelVariants(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]], columns=['hour', 'y'])

        def rooney_biegler_params(data):
            model = pyo.ConcreteModel()
            model.asymptote = pyo.Param(initialize=15, mutable=True)
            model.rate_constant = pyo.Param(initialize=0.5, mutable=True)

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr
            model.response_function = pyo.Expression(data.hour, rule=response_rule)
            return model

        def rooney_biegler_indexed_params(data):
            model = pyo.ConcreteModel()
            model.param_names = pyo.Set(initialize=['asymptote', 'rate_constant'])
            model.theta = pyo.Param(model.param_names, initialize={'asymptote': 15, 'rate_constant': 0.5}, mutable=True)

            def response_rule(m, h):
                expr = m.theta['asymptote'] * (1 - pyo.exp(-m.theta['rate_constant'] * h))
                return expr
            model.response_function = pyo.Expression(data.hour, rule=response_rule)
            return model

        def rooney_biegler_vars(data):
            model = pyo.ConcreteModel()
            model.asymptote = pyo.Var(initialize=15)
            model.rate_constant = pyo.Var(initialize=0.5)
            model.asymptote.fixed = True
            model.rate_constant.fixed = True

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr
            model.response_function = pyo.Expression(data.hour, rule=response_rule)
            return model

        def rooney_biegler_indexed_vars(data):
            model = pyo.ConcreteModel()
            model.var_names = pyo.Set(initialize=['asymptote', 'rate_constant'])
            model.theta = pyo.Var(model.var_names, initialize={'asymptote': 15, 'rate_constant': 0.5})
            model.theta['asymptote'].fixed = True
            model.theta['rate_constant'].fixed = True

            def response_rule(m, h):
                expr = m.theta['asymptote'] * (1 - pyo.exp(-m.theta['rate_constant'] * h))
                return expr
            model.response_function = pyo.Expression(data.hour, rule=response_rule)
            return model

        def SSE(model, data):
            expr = sum(((data.y[i] - model.response_function[data.hour[i]]) ** 2 for i in data.index))
            return expr
        self.objective_function = SSE
        theta_vals = pd.DataFrame([20, 1], index=['asymptote', 'rate_constant']).T
        theta_vals_index = pd.DataFrame([20, 1], index=["theta['asymptote']", "theta['rate_constant']"]).T
        self.input = {'param': {'model': rooney_biegler_params, 'theta_names': ['asymptote', 'rate_constant'], 'theta_vals': theta_vals}, 'param_index': {'model': rooney_biegler_indexed_params, 'theta_names': ['theta'], 'theta_vals': theta_vals_index}, 'vars': {'model': rooney_biegler_vars, 'theta_names': ['asymptote', 'rate_constant'], 'theta_vals': theta_vals}, 'vars_index': {'model': rooney_biegler_indexed_vars, 'theta_names': ['theta'], 'theta_vals': theta_vals_index}, 'vars_quoted_index': {'model': rooney_biegler_indexed_vars, 'theta_names': ["theta['asymptote']", "theta['rate_constant']"], 'theta_vals': theta_vals_index}, 'vars_str_index': {'model': rooney_biegler_indexed_vars, 'theta_names': ['theta[asymptote]', 'theta[rate_constant]'], 'theta_vals': theta_vals_index}}

    @unittest.skipIf(not pynumero_ASL_available, 'pynumero ASL is not available')
    @unittest.skipIf(not parmest.inverse_reduced_hessian_available, 'Cannot test covariance matrix: required ASL dependency is missing')
    def test_parmest_basics(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(parmest_input['model'], self.data, parmest_input['theta_names'], self.objective_function)
            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
            self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 1], 0.04193591, places=2)
            obj_at_theta = pest.objective_at_theta(parmest_input['theta_vals'])
            self.assertAlmostEqual(obj_at_theta['obj'][0], 16.531953, places=2)

    def test_parmest_basics_with_initialize_parmest_model_option(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(parmest_input['model'], self.data, parmest_input['theta_names'], self.objective_function)
            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
            self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 1], 0.04193591, places=2)
            obj_at_theta = pest.objective_at_theta(parmest_input['theta_vals'], initialize_parmest_model=True)
            self.assertAlmostEqual(obj_at_theta['obj'][0], 16.531953, places=2)

    def test_parmest_basics_with_square_problem_solve(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(parmest_input['model'], self.data, parmest_input['theta_names'], self.objective_function)
            obj_at_theta = pest.objective_at_theta(parmest_input['theta_vals'], initialize_parmest_model=True)
            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
            self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 1], 0.04193591, places=2)
            self.assertAlmostEqual(obj_at_theta['obj'][0], 16.531953, places=2)

    def test_parmest_basics_with_square_problem_solve_no_theta_vals(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(parmest_input['model'], self.data, parmest_input['theta_names'], self.objective_function)
            obj_at_theta = pest.objective_at_theta(initialize_parmest_model=True)
            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
            self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
            self.assertAlmostEqual(cov.iloc[1, 1], 0.04193591, places=2)