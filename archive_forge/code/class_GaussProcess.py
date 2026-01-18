import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt
class GaussProcess:
    """class to perform kernel ridge regression (gaussian process)

    Warning: this class is memory intensive, it creates nobs x nobs distance
    matrix and its inverse, where nobs is the number of rows (observations).
    See sparse version for larger number of observations


    Notes
    -----

    Todo:
    * normalize multidimensional x array on demand, either by var or cov
    * add confidence band
    * automatic selection or proposal of smoothing parameters

    Note: this is different from kernel smoothing regression,
       see for example https://en.wikipedia.org/wiki/Kernel_smoother

    In this version of the kernel ridge regression, the training points
    are fitted exactly.
    Needs a fast version for leave-one-out regression, for fitting each
    observation on all the other points.
    This version could be numerically improved for the calculation for many
    different values of the ridge coefficient. see also short summary by
    Isabelle Guyon (ETHZ) in a manuscript KernelRidge.pdf

    Needs verification and possibly additional statistical results or
    summary statistics for interpretation, but this is a problem with
    non-parametric, non-linear methods.

    Reference
    ---------

    Rasmussen, C.E. and C.K.I. Williams, 2006, Gaussian Processes for Machine
    Learning, the MIT Press, www.GaussianProcess.org/gpal, chapter 2

    a short summary of the kernel ridge regression is at
    http://www.ics.uci.edu/~welling/teaching/KernelsICS273B/Kernel-Ridge.pdf
    """

    def __init__(self, x, y=None, kernel=kernel_rbf, scale=0.5, ridgecoeff=1e-10, **kwds):
        """
        Parameters
        ----------
        x : 2d array (N,K)
           data array of explanatory variables, columns represent variables
           rows represent observations
        y : 2d array (N,1) (optional)
           endogenous variable that should be fitted or predicted
           can alternatively be specified as parameter to fit method
        kernel : function, default: kernel_rbf
           kernel: (x1,x2)->kernel matrix is a function that takes as parameter
           two column arrays and return the kernel or distance matrix
        scale : float (optional)
           smoothing parameter for the rbf kernel
        ridgecoeff : float (optional)
           coefficient that is multiplied with the identity matrix in the
           ridge regression

        Notes
        -----
        After initialization, kernel matrix is calculated and if y is given
        as parameter then also the linear regression parameter and the
        fitted or estimated y values, yest, are calculated. yest is available
        as an attribute in this case.

        Both scale and the ridge coefficient smooth the fitted curve.

        """
        self.x = x
        self.kernel = kernel
        self.scale = scale
        self.ridgecoeff = ridgecoeff
        self.distxsample = kernel(x, x, scale=scale)
        self.Kinv = np.linalg.inv(self.distxsample + np.eye(*self.distxsample.shape) * ridgecoeff)
        if y is not None:
            self.y = y
            self.yest = self.fit(y)

    def fit(self, y):
        """fit the training explanatory variables to a sample ouput variable"""
        self.parest = np.dot(self.Kinv, y)
        yhat = np.dot(self.distxsample, self.parest)
        return yhat

    def predict(self, x):
        """predict new y values for a given array of explanatory variables"""
        self.xpredict = x
        distxpredict = self.kernel(x, self.x, scale=self.scale)
        self.ypredict = np.dot(distxpredict, self.parest)
        return self.ypredict

    def plot(self, y, plt=plt):
        """some basic plots"""
        plt.figure()
        plt.plot(self.x, self.y, 'bo-', self.x, self.yest, 'r.-')
        plt.title('sample (training) points')
        plt.figure()
        plt.plot(self.xpredict, y, 'bo-', self.xpredict, self.ypredict, 'r.-')
        plt.title('all points')