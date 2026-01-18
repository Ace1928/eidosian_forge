import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf
class MVElliptical:
    """Base Class for multivariate elliptical distributions, normal and t

    contains common initialization, and some common methods
    subclass needs to implement at least rvs and logpdf methods

    """

    def __init__(self, mean, sigma, *args, **kwds):
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
        self.extra_args = []
        self.mean = np.asarray(mean)
        self.sigma = sigma = np.asarray(sigma)
        sigma = np.squeeze(sigma)
        self.nvars = nvars = len(mean)
        if sigma.shape == ():
            self.sigma = np.eye(nvars) * sigma
            self.sigmainv = np.eye(nvars) / sigma
            self.cholsigmainv = np.eye(nvars) / np.sqrt(sigma)
        elif sigma.ndim == 1 and len(sigma) == nvars:
            self.sigma = np.diag(sigma)
            self.sigmainv = np.diag(1.0 / sigma)
            self.cholsigmainv = np.diag(1.0 / np.sqrt(sigma))
        elif sigma.shape == (nvars, nvars):
            self.sigmainv = np.linalg.pinv(sigma)
            self.cholsigmainv = np.linalg.cholesky(self.sigmainv).T
        else:
            raise ValueError('sigma has invalid shape')
        self.logdetsigma = np.log(np.linalg.det(self.sigma))

    def rvs(self, size=1):
        """random variable

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


        """
        raise NotImplementedError

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


        this should be made to work with 2d x,
        with multivariate normal vector in each row and iid across rows
        does not work now because of dot in whiten

        """
        raise NotImplementedError

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
        raise NotImplementedError

    def affine_transformed(self, shift, scale_matrix):
        """affine transformation define in subclass because of distribution
        specific restrictions"""
        raise NotImplementedError

    def whiten(self, x):
        """
        whiten the data by linear transformation

        Parameters
        ----------
        x : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x, self.cholsigmainv.T)

        Notes
        -----
        This only does rescaling, it does not subtract the mean, use standardize
        for this instead

        See Also
        --------
        standardize : subtract mean and rescale to standardized random variable.
        """
        x = np.asarray(x)
        return np.dot(x, self.cholsigmainv.T)

    def pdf(self, x):
        """probability density function

        Parameters
        ----------
        x : array_like
            can be 1d or 2d, if 2d, then each row is taken as independent
            multivariate random vector

        Returns
        -------
        pdf : float or array
            probability density value of each random vector

        """
        return np.exp(self.logpdf(x))

    def standardize(self, x):
        """standardize the random variable, i.e. subtract mean and whiten

        Parameters
        ----------
        x : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        np.dot(x - self.mean, self.cholsigmainv.T)

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        """
        return self.whiten(x - self.mean)

    def standardized(self):
        """return new standardized MVNormal instance
        """
        return self.affine_transformed(-self.mean, self.cholsigmainv)

    def normalize(self, x):
        """normalize the random variable, i.e. subtract mean and rescale

        The distribution will have zero mean and sigma equal to correlation

        Parameters
        ----------
        x : array_like, 1d or 2d
            Data to be whitened, if 2d then each row contains an independent
            sample of the multivariate random vector

        Returns
        -------
        (x - self.mean)/std_sigma

        Notes
        -----


        See Also
        --------
        whiten : rescale random variable, standardize without subtracting mean.


        """
        std_ = np.atleast_2d(self.std_sigma)
        return (x - self.mean) / std_

    def normalized(self, demeaned=True):
        """return a normalized distribution where sigma=corr

        if demeaned is True, then mean will be set to zero

        """
        if demeaned:
            mean_new = np.zeros_like(self.mean)
        else:
            mean_new = self.mean / self.std_sigma
        sigma_new = self.corr
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)

    def normalized2(self, demeaned=True):
        """return a normalized distribution where sigma=corr



        second implementation for testing affine transformation
        """
        if demeaned:
            shift = -self.mean
        else:
            shift = self.mean * (1.0 / self.std_sigma - 1.0)
        return self.affine_transformed(shift, np.diag(1.0 / self.std_sigma))

    @property
    def std(self):
        """standard deviation, square root of diagonal elements of cov
        """
        return np.sqrt(np.diag(self.cov))

    @property
    def std_sigma(self):
        """standard deviation, square root of diagonal elements of sigma
        """
        return np.sqrt(np.diag(self.sigma))

    @property
    def corr(self):
        """correlation matrix"""
        return self.cov / np.outer(self.std, self.std)
    expect_mc = expect_mc

    def marginal(self, indices):
        """return marginal distribution for variables given by indices

        this should be correct for normal and t distribution

        Parameters
        ----------
        indices : array_like, int
            list of indices of variables in the marginal distribution

        Returns
        -------
        mvdist : instance
            new instance of the same multivariate distribution class that
            contains the marginal distribution of the variables given in
            indices

        """
        indices = np.asarray(indices)
        mean_new = self.mean[indices]
        sigma_new = self.sigma[indices[:, None], indices]
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)