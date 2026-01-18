import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
class MVT(MVElliptical):
    __name__ == 'Multivariate Student T Distribution'

    def __init__(self, mean, sigma, df):
        """initialize instance

        Parameters
        ----------
        mean : array_like
            parameter mu (might be renamed), for symmetric distributions this
            is the mean
        sigma : array_like, 2d
            dispersion matrix, covariance matrix in normal distribution, but
            only proportional to covariance matrix in t distribution
        args : list
            distribution specific arguments, e.g. df for t distribution
        kwds : dict
            currently not used

        """
        super().__init__(mean, sigma)
        self.extra_args = ['df']
        self.df = df

    def rvs(self, size=1):
        """random variables with Student T distribution

        Parameters
        ----------
        size : int or tuple
            the number and shape of random variables to draw.

        Returns
        -------
        rvs : ndarray
            the returned random variables with shape given by size and the
            dimension of the multivariate random vector as additional last
            dimension
            - TODO: Not sure if this works for size tuples with len>1.

        Notes
        -----
        generated as a chi-square mixture of multivariate normal random
        variables.
        does this require df>2 ?


        """
        from .multivariate import multivariate_t_rvs
        return multivariate_t_rvs(self.mean, self.sigma, df=self.df, n=size)

    def logpdf(self, x):
        """logarithm of probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        logpdf : float or array
            probability density value of each random vector

        """
        x = np.asarray(x)
        df = self.df
        nvars = self.nvars
        x_whitened = self.whiten(x - self.mean)
        llf = -nvars * np_log(df * np_pi)
        llf -= self.logdetsigma
        llf -= (df + nvars) * np_log(1 + np.sum(x_whitened ** 2, -1) / df)
        llf *= 0.5
        llf += sps_gamln((df + nvars) / 2.0) - sps_gamln(df / 2.0)
        return llf

    def cdf(self, x, **kwds):
        """cumulative distribution function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector
        kwds : dict
            contains options for the numerical calculation of the cdf

        Returns
        -------
        cdf : float or array
            probability density value of each random vector

        """
        lower = -np.inf * np.ones_like(x)
        upper = (x - self.mean) / self.std_sigma
        return mvstdtprob(lower, upper, self.corr, self.df, **kwds)

    @property
    def cov(self):
        """covariance matrix

        The covariance matrix for the t distribution does not exist for df<=2,
        and is equal to sigma * df/(df-2) for df>2

        """
        if self.df <= 2:
            return np.nan * np.ones_like(self.sigma)
        else:
            return self.df / (self.df - 2.0) * self.sigma

    def affine_transformed(self, shift, scale_matrix):
        """return distribution of a full rank affine transform

        for full rank scale_matrix only

        Parameters
        ----------
        shift : array_like
            shift of mean
        scale_matrix : array_like
            linear transformation matrix

        Returns
        -------
        mvt : instance of MVT
            instance of multivariate t distribution given by affine
            transformation


        Notes
        -----

        This checks for eigvals<=0, so there are possible problems for cases
        with positive eigenvalues close to zero.

        see: http://www.statlect.com/mcdstu1.htm

        I'm not sure about general case, non-full rank transformation are not
        multivariate t distributed.

        y = a + B x

        where a is shift,
        B is full rank scale matrix with same dimension as sigma

        """
        B = scale_matrix
        if not B.shape == (self.nvars, self.nvars):
            if (np.linalg.eigvals(B) <= 0).any():
                raise ValueError('affine transform has to be full rank')
        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVT(mean_new, sigma_new, self.df)