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
class TestRooneyBiegler(unittest.TestCase):

    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model
        data = pd.DataFrame(data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]], columns=['hour', 'y'])
        theta_names = ['asymptote', 'rate_constant']

        def SSE(model, data):
            expr = sum(((data.y[i] - model.response_function[data.hour[i]]) ** 2 for i in data.index))
            return expr
        solver_options = {'tol': 1e-08}
        self.data = data
        self.pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE, solver_options=solver_options, tee=True)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()
        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.1426, places=2)
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5311, places=2)

    @unittest.skipIf(not graphics.imports_available, 'parmest.graphics imports are unavailable')
    def test_bootstrap(self):
        objval, thetavals = self.pest.theta_est()
        num_bootstraps = 10
        theta_est = self.pest.theta_est_bootstrap(num_bootstraps, return_samples=True)
        num_samples = theta_est['samples'].apply(len)
        self.assertTrue(len(theta_est.index), 10)
        self.assertTrue(num_samples.equals(pd.Series([6] * 10)))
        del theta_est['samples']
        CR = self.pest.confidence_region_test(theta_est, 'MVN', [0.5, 0.75, 1.0])
        self.assertTrue(set(CR.columns) >= set([0.5, 0.75, 1.0]))
        self.assertTrue(CR[0.5].sum() == 5)
        self.assertTrue(CR[0.75].sum() == 7)
        self.assertTrue(CR[1.0].sum() == 10)
        graphics.pairwise_plot(theta_est)
        graphics.pairwise_plot(theta_est, thetavals)
        graphics.pairwise_plot(theta_est, thetavals, 0.8, ['MVN', 'KDE', 'Rect'])

    @unittest.skipIf(not graphics.imports_available, 'parmest.graphics imports are unavailable')
    def test_likelihood_ratio(self):
        objval, thetavals = self.pest.theta_est()
        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)
        obj_at_theta = self.pest.objective_at_theta(theta_vals)
        LR = self.pest.likelihood_ratio_test(obj_at_theta, objval, [0.8, 0.9, 1.0])
        self.assertTrue(set(LR.columns) >= set([0.8, 0.9, 1.0]))
        self.assertTrue(LR[0.8].sum() == 6)
        self.assertTrue(LR[0.9].sum() == 10)
        self.assertTrue(LR[1.0].sum() == 60)
        graphics.pairwise_plot(LR, thetavals, 0.8)

    def test_leaveNout(self):
        lNo_theta = self.pest.theta_est_leaveNout(1)
        self.assertTrue(lNo_theta.shape == (6, 2))
        results = self.pest.leaveNout_bootstrap_test(1, None, 3, 'Rect', [0.5, 1.0], seed=5436)
        self.assertTrue(len(results) == 6)
        i = 1
        samples = results[i][0]
        lno_theta = results[i][1]
        bootstrap_theta = results[i][2]
        self.assertTrue(samples == [1])
        self.assertTrue(lno_theta.shape[0] == 1)
        self.assertTrue(set(lno_theta.columns) >= set([0.5, 1.0]))
        self.assertTrue(lno_theta[1.0].sum() == 1)
        self.assertTrue(bootstrap_theta.shape[0] == 3)
        self.assertTrue(bootstrap_theta[1.0].sum() == 3)

    def test_diagnostic_mode(self):
        self.pest.diagnostic_mode = True
        objval, thetavals = self.pest.theta_est()
        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)
        obj_at_theta = self.pest.objective_at_theta(theta_vals)
        self.pest.diagnostic_mode = False

    @unittest.skip('Presently having trouble with mpiexec on appveyor')
    def test_parallel_parmest(self):
        """use mpiexec and mpi4py"""
        p = str(parmestbase.__path__)
        l = p.find("'")
        r = p.find("'", l + 1)
        parmestpath = p[l + 1:r]
        rbpath = parmestpath + os.sep + 'examples' + os.sep + 'rooney_biegler' + os.sep + 'rooney_biegler_parmest.py'
        rbpath = os.path.abspath(rbpath)
        rlist = ['mpiexec', '--allow-run-as-root', '-n', '2', sys.executable, rbpath]
        if sys.version_info >= (3, 5):
            ret = subprocess.run(rlist)
            retcode = ret.returncode
        else:
            retcode = subprocess.call(rlist)
        assert retcode == 0

    @unittest.skip("Most folks don't have k_aug installed")
    def test_theta_k_aug_for_Hessian(self):
        objval, thetavals, Hessian = self.pest.theta_est(solver='k_aug')
        self.assertAlmostEqual(objval, 4.4675, places=2)

    @unittest.skipIf(not pynumero_ASL_available, 'pynumero ASL is not available')
    @unittest.skipIf(not parmest.inverse_reduced_hessian_available, 'Cannot test covariance matrix: required ASL dependency is missing')
    def test_theta_est_cov(self):
        objval, thetavals, cov = self.pest.theta_est(calc_cov=True, cov_n=6)
        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.1426, places=2)
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5311, places=2)
        self.assertAlmostEqual(cov.iloc[0, 0], 6.30579403, places=2)
        self.assertAlmostEqual(cov.iloc[0, 1], -0.4395341, places=2)
        self.assertAlmostEqual(cov.iloc[1, 0], -0.4395341, places=2)
        self.assertAlmostEqual(cov.iloc[1, 1], 0.04124, places=2)
        ' Why does the covariance matrix from parmest not match the paper? Parmest is\n        calculating the exact reduced Hessian. The paper (Rooney and Bielger, 2001) likely\n        employed the first order approximation common for nonlinear regression. The paper\n        values were verified with Scipy, which uses the same first order approximation.\n        The formula used in parmest was verified against equations (7-5-15) and (7-5-16) in\n        "Nonlinear Parameter Estimation", Y. Bard, 1974.\n        '

    def test_cov_scipy_least_squares_comparison(self):
        """
        Scipy results differ in the 3rd decimal place from the paper. It is possible
        the paper used an alternative finite difference approximation for the Jacobian.
        """

        def model(theta, t):
            """
            Model to be fitted y = model(theta, t)
            Arguments:
                theta: vector of fitted parameters
                t: independent variable [hours]

            Returns:
                y: model predictions [need to check paper for units]
            """
            asymptote = theta[0]
            rate_constant = theta[1]
            return asymptote * (1 - np.exp(-rate_constant * t))

        def residual(theta, t, y):
            """
            Calculate residuals
            Arguments:
                theta: vector of fitted parameters
                t: independent variable [hours]
                y: dependent variable [?]
            """
            return y - model(theta, t)
        t = self.data['hour'].to_numpy()
        y = self.data['y'].to_numpy()
        theta_guess = np.array([15, 0.5])
        sol = scipy.optimize.least_squares(residual, theta_guess, method='trf', args=(t, y), verbose=2)
        theta_hat = sol.x
        self.assertAlmostEqual(theta_hat[0], 19.1426, places=2)
        self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)
        r = residual(theta_hat, t, y)
        sigre = np.matmul(r.T, r / (len(y) - 2))
        cov = sigre * np.linalg.inv(np.matmul(sol.jac.T, sol.jac))
        self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)
        self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)
        self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)
        self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)

    def test_cov_scipy_curve_fit_comparison(self):
        """
        Scipy results differ in the 3rd decimal place from the paper. It is possible
        the paper used an alternative finite difference approximation for the Jacobian.
        """

        def model(t, asymptote, rate_constant):
            return asymptote * (1 - np.exp(-rate_constant * t))
        t = self.data['hour'].to_numpy()
        y = self.data['y'].to_numpy()
        theta_guess = np.array([15, 0.5])
        theta_hat, cov = scipy.optimize.curve_fit(model, t, y, p0=theta_guess)
        self.assertAlmostEqual(theta_hat[0], 19.1426, places=2)
        self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)
        self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)
        self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)
        self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)
        self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)