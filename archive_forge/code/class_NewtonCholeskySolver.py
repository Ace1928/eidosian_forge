import warnings
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.optimize
from ..._loss.loss import HalfSquaredError
from ...exceptions import ConvergenceWarning
from ...utils.optimize import _check_optimize_result
from .._linear_loss import LinearModelLoss
class NewtonCholeskySolver(NewtonSolver):
    """Cholesky based Newton solver.

    Inner solver for finding the Newton step H w_newton = -g uses Cholesky based linear
    solver.
    """

    def setup(self, X, y, sample_weight):
        super().setup(X=X, y=y, sample_weight=sample_weight)
        n_dof = X.shape[1]
        if self.linear_loss.fit_intercept:
            n_dof += 1
        self.gradient = np.empty_like(self.coef)
        self.hessian = np.empty_like(self.coef, shape=(n_dof, n_dof))

    def update_gradient_hessian(self, X, y, sample_weight):
        _, _, self.hessian_warning = self.linear_loss.gradient_hessian(coef=self.coef, X=X, y=y, sample_weight=sample_weight, l2_reg_strength=self.l2_reg_strength, n_threads=self.n_threads, gradient_out=self.gradient, hessian_out=self.hessian, raw_prediction=self.raw_prediction)

    def inner_solve(self, X, y, sample_weight):
        if self.hessian_warning:
            warnings.warn(f'The inner solver of {self.__class__.__name__} detected a pointwise hessian with many negative values at iteration #{self.iteration}. It will now resort to lbfgs instead.', ConvergenceWarning)
            if self.verbose:
                print('  The inner solver detected a pointwise Hessian with many negative values and resorts to lbfgs instead.')
            self.use_fallback_lbfgs_solve = True
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', scipy.linalg.LinAlgWarning)
                self.coef_newton = scipy.linalg.solve(self.hessian, -self.gradient, check_finite=False, assume_a='sym')
                self.gradient_times_newton = self.gradient @ self.coef_newton
                if self.gradient_times_newton > 0:
                    if self.verbose:
                        print('  The inner solver found a Newton step that is not a descent direction and resorts to LBFGS steps instead.')
                    self.use_fallback_lbfgs_solve = True
                    return
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgWarning) as e:
            warnings.warn(f'The inner solver of {self.__class__.__name__} stumbled upon a singular or very ill-conditioned Hessian matrix at iteration #{self.iteration}. It will now resort to lbfgs instead.\nFurther options are to use another solver or to avoid such situation in the first place. Possible remedies are removing collinear features of X or increasing the penalization strengths.\nThe original Linear Algebra message was:\n' + str(e), scipy.linalg.LinAlgWarning)
            if self.verbose:
                print('  The inner solver stumbled upon an singular or ill-conditioned Hessian matrix and resorts to LBFGS instead.')
            self.use_fallback_lbfgs_solve = True
            return