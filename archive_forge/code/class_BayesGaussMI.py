import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
class BayesGaussMI:
    """
    Bayesian Imputation using a Gaussian model.

    The approach is Bayesian.  The goal is to sample from the joint
    distribution of the mean vector, covariance matrix, and missing
    data values given the observed data values.  Conjugate priors for
    the population mean and covariance matrix are used.  Gibbs
    sampling is used to update the mean vector, covariance matrix, and
    missing data values in turn.  After burn-in, the imputed complete
    data sets from the Gibbs chain can be used in multiple imputation
    analyses (MI).

    Parameters
    ----------
    data : ndarray
        The array of data to be imputed.  Values in the array equal to
        NaN are imputed.
    mean_prior : ndarray, optional
        The covariance matrix of the Gaussian prior distribution for
        the mean vector.  If not provided, the identity matrix is
        used.
    cov_prior : ndarray, optional
        The center matrix for the inverse Wishart prior distribution
        for the covariance matrix.  If not provided, the identity
        matrix is used.
    cov_prior_df : positive float
        The degrees of freedom of the inverse Wishart prior
        distribution for the covariance matrix.  Defaults to 1.

    Examples
    --------
    A basic example with OLS. Data is generated assuming 10% is missing at
    random.

    >>> import numpy as np
    >>> x = np.random.standard_normal((1000, 2))
    >>> x.flat[np.random.sample(2000) < 0.1] = np.nan

    The imputer is used with ``MI``.

    >>> import statsmodels.api as sm
    >>> def model_args_fn(x):
    ...     # Return endog, exog from x
    ...    return x[:, 0], x[:, 1:]
    >>> imp = sm.BayesGaussMI(x)
    >>> mi = sm.MI(imp, sm.OLS, model_args_fn)
    """

    def __init__(self, data, mean_prior=None, cov_prior=None, cov_prior_df=1):
        self.exog_names = None
        if type(data) is pd.DataFrame:
            self.exog_names = data.columns
        data = np.require(data, requirements='W')
        self.data = data
        self._data = data
        self.mask = np.isnan(data)
        self.nobs = self.mask.shape[0]
        self.nvar = self.mask.shape[1]
        z = 1 + np.log(1 + np.arange(self.mask.shape[1]))
        c = np.dot(self.mask, z)
        rowmap = {}
        for i, v in enumerate(c):
            if v == 0:
                continue
            if v not in rowmap:
                rowmap[v] = []
            rowmap[v].append(i)
        self.patterns = [np.asarray(v) for v in rowmap.values()]
        p = self._data.shape[1]
        self.cov = np.eye(p)
        mean = []
        for i in range(p):
            v = self._data[:, i]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                msg = 'Column %d has no observed values' % i
                raise ValueError(msg)
            mean.append(v.mean())
        self.mean = np.asarray(mean)
        if mean_prior is None:
            mean_prior = np.eye(p)
        self.mean_prior = mean_prior
        if cov_prior is None:
            cov_prior = np.eye(p)
        self.cov_prior = cov_prior
        self.cov_prior_df = cov_prior_df

    def update(self):
        """
        Cycle through all Gibbs updates.
        """
        self.update_data()
        self.update_mean()
        self.update_cov()

    def update_data(self):
        """
        Gibbs update of the missing data values.
        """
        for ix in self.patterns:
            i = ix[0]
            ix_miss = np.flatnonzero(self.mask[i, :])
            ix_obs = np.flatnonzero(~self.mask[i, :])
            mm = self.mean[ix_miss]
            mo = self.mean[ix_obs]
            voo = self.cov[ix_obs, :][:, ix_obs]
            vmm = self.cov[ix_miss, :][:, ix_miss]
            vmo = self.cov[ix_miss, :][:, ix_obs]
            r = self._data[ix, :][:, ix_obs] - mo
            cm = mm + np.dot(vmo, np.linalg.solve(voo, r.T)).T
            cv = vmm - np.dot(vmo, np.linalg.solve(voo, vmo.T))
            cs = np.linalg.cholesky(cv)
            u = np.random.normal(size=(len(ix), len(ix_miss)))
            self._data[np.ix_(ix, ix_miss)] = cm + np.dot(u, cs.T)
        if self.exog_names is not None:
            self.data = pd.DataFrame(self._data, columns=self.exog_names, copy=False)
        else:
            self.data = self._data

    def update_mean(self):
        """
        Gibbs update of the mean vector.

        Do not call until update_data has been called once.
        """
        cm = np.linalg.solve(self.cov / self.nobs + self.mean_prior, self.mean_prior / self.nobs)
        cm = np.dot(self.cov, cm)
        vm = np.linalg.solve(self.cov, self._data.sum(0))
        vm = np.dot(cm, vm)
        r = np.linalg.cholesky(cm)
        self.mean = vm + np.dot(r, np.random.normal(0, 1, self.nvar))

    def update_cov(self):
        """
        Gibbs update of the covariance matrix.

        Do not call until update_data has been called once.
        """
        r = self._data - self.mean
        gr = np.dot(r.T, r)
        a = gr + self.cov_prior
        df = int(np.ceil(self.nobs + self.cov_prior_df))
        r = np.linalg.cholesky(np.linalg.inv(a))
        x = np.dot(np.random.normal(size=(df, self.nvar)), r.T)
        ma = np.dot(x.T, x)
        self.cov = np.linalg.inv(ma)