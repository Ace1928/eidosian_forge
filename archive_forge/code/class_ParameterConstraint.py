from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
class ParameterConstraint:
    """
    A class for managing linear equality constraints for a parameter
    vector.
    """

    def __init__(self, lhs, rhs, exog):
        """
        Parameters
        ----------
        lhs : ndarray
           A q x p matrix which is the left hand side of the
           constraint lhs * param = rhs.  The number of constraints is
           q >= 1 and p is the dimension of the parameter vector.
        rhs : ndarray
          A 1-dimensional vector of length q which is the right hand
          side of the constraint equation.
        exog : ndarray
          The n x p exognenous data for the full model.
        """
        rhs = np.atleast_1d(rhs.squeeze())
        if rhs.ndim > 1:
            raise ValueError('The right hand side of the constraint must be a vector.')
        if len(rhs) != lhs.shape[0]:
            raise ValueError('The number of rows of the left hand side constraint matrix L must equal the length of the right hand side constraint vector R.')
        self.lhs = lhs
        self.rhs = rhs
        lhs_u, lhs_s, lhs_vt = np.linalg.svd(lhs.T, full_matrices=1)
        self.lhs0 = lhs_u[:, len(lhs_s):]
        self.lhs1 = lhs_u[:, 0:len(lhs_s)]
        self.lhsf = np.hstack((self.lhs0, self.lhs1))
        self.param0 = np.dot(self.lhs1, np.dot(lhs_vt, self.rhs) / lhs_s)
        self._offset_increment = np.dot(exog, self.param0)
        self.orig_exog = exog
        self.exog_fulltrans = np.dot(exog, self.lhsf)

    def offset_increment(self):
        """
        Returns a vector that should be added to the offset vector to
        accommodate the constraint.

        Parameters
        ----------
        exog : array_like
           The exogeneous data for the model.
        """
        return self._offset_increment

    def reduced_exog(self):
        """
        Returns a linearly transformed exog matrix whose columns span
        the constrained model space.

        Parameters
        ----------
        exog : array_like
           The exogeneous data for the model.
        """
        return self.exog_fulltrans[:, 0:self.lhs0.shape[1]]

    def restore_exog(self):
        """
        Returns the full exog matrix before it was reduced to
        satisfy the constraint.
        """
        return self.orig_exog

    def unpack_param(self, params):
        """
        Converts the parameter vector `params` from reduced to full
        coordinates.
        """
        return self.param0 + np.dot(self.lhs0, params)

    def unpack_cov(self, bcov):
        """
        Converts the covariance matrix `bcov` from reduced to full
        coordinates.
        """
        return np.dot(self.lhs0, np.dot(bcov, self.lhs0.T))