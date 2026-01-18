import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class CovarianceReduction(_DimReductionRegression):
    """
    Dimension reduction for covariance matrices (CORE).

    Parameters
    ----------
    endog : array_like
        The dependent variable, treated as group labels
    exog : array_like
        The independent variables.
    dim : int
        The dimension of the subspace onto which the covariance
        matrices are projected.

    Returns
    -------
    A model instance.  Call `fit` on the model instance to obtain
    a results instance, which contains the fitted model parameters.

    Notes
    -----
    This is a likelihood-based dimension reduction procedure based
    on Wishart models for sample covariance matrices.  The goal
    is to find a projection matrix P so that C_i | P'C_iP and
    C_j | P'C_jP are equal in distribution for all i, j, where
    the C_i are the within-group covariance matrices.

    The model and methodology are as described in Cook and Forzani.
    The optimization method follows Edelman et. al.

    References
    ----------
    DR Cook, L Forzani (2008).  Covariance reducing models: an alternative
    to spectral modeling of covariance matrices.  Biometrika 95:4.

    A Edelman, TA Arias, ST Smith (1998).  The geometry of algorithms with
    orthogonality constraints. SIAM J Matrix Anal Appl.
    http://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
    """

    def __init__(self, endog, exog, dim):
        super().__init__(endog, exog)
        covs, ns = ([], [])
        df = pd.DataFrame(self.exog, index=self.endog)
        for _, v in df.groupby(df.index):
            covs.append(v.cov().values)
            ns.append(v.shape[0])
        self.nobs = len(endog)
        covm = 0
        for i, _ in enumerate(covs):
            covm += covs[i] * ns[i]
        covm /= self.nobs
        self.covm = covm
        self.covs = covs
        self.ns = ns
        self.dim = dim

    def loglike(self, params):
        """
        Evaluate the log-likelihood

        Parameters
        ----------
        params : array_like
            The projection matrix used to reduce the covariances, flattened
            to 1d.

        Returns the log-likelihood.
        """
        p = self.covm.shape[0]
        proj = params.reshape((p, self.dim))
        c = np.dot(proj.T, np.dot(self.covm, proj))
        _, ldet = np.linalg.slogdet(c)
        f = self.nobs * ldet / 2
        for j, c in enumerate(self.covs):
            c = np.dot(proj.T, np.dot(c, proj))
            _, ldet = np.linalg.slogdet(c)
            f -= self.ns[j] * ldet / 2
        return f

    def score(self, params):
        """
        Evaluate the score function.

        Parameters
        ----------
        params : array_like
            The projection matrix used to reduce the covariances,
            flattened to 1d.

        Returns the score function evaluated at 'params'.
        """
        p = self.covm.shape[0]
        proj = params.reshape((p, self.dim))
        c0 = np.dot(proj.T, np.dot(self.covm, proj))
        cP = np.dot(self.covm, proj)
        g = self.nobs * np.linalg.solve(c0, cP.T).T
        for j, c in enumerate(self.covs):
            c0 = np.dot(proj.T, np.dot(c, proj))
            cP = np.dot(c, proj)
            g -= self.ns[j] * np.linalg.solve(c0, cP.T).T
        return g.ravel()

    def fit(self, start_params=None, maxiter=200, gtol=0.0001):
        """
        Fit the covariance reduction model.

        Parameters
        ----------
        start_params : array_like
            Starting value for the projection matrix. May be
            rectangular, or flattened.
        maxiter : int
            The maximum number of gradient steps to take.
        gtol : float
            Convergence criterion for the gradient norm.

        Returns
        -------
        A results instance that can be used to access the
        fitted parameters.
        """
        p = self.covm.shape[0]
        d = self.dim
        if start_params is None:
            params = np.zeros((p, d))
            params[0:d, 0:d] = np.eye(d)
            params = params
        else:
            params = start_params
        params, llf, cnvrg = _grass_opt(params, lambda x: -self.loglike(x), lambda x: -self.score(x), maxiter, gtol)
        llf *= -1
        if not cnvrg:
            g = self.score(params.ravel())
            gn = np.sqrt(np.sum(g * g))
            msg = 'CovReduce optimization did not converge, |g|=%f' % gn
            warnings.warn(msg, ConvergenceWarning)
        results = DimReductionResults(self, params, eigs=None)
        results.llf = llf
        return DimReductionResultsWrapper(results)