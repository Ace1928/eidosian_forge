import warnings
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.optimize
from ..._loss.loss import HalfSquaredError
from ...exceptions import ConvergenceWarning
from ...utils.optimize import _check_optimize_result
from .._linear_loss import LinearModelLoss
Solve the optimization problem.

        This is the main routine.

        Order of calls:
            self.setup()
            while iteration:
                self.update_gradient_hessian()
                self.inner_solve()
                self.line_search()
                self.check_convergence()
            self.finalize()

        Returns
        -------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Solution of the optimization problem.
        