import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
class VarmaPoly:
    """class to keep track of Varma polynomial format


    Examples
    --------

    ar23 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[-0.6,  0. ],
                     [ 0.2, -0.6]],

                    [[-0.1,  0. ],
                     [ 0.1, -0.1]]])

    ma22 = np.array([[[ 1. ,  0. ],
                     [ 0. ,  1. ]],

                    [[ 0.4,  0. ],
                     [ 0.2, 0.3]]])


    """

    def __init__(self, ar, ma=None):
        self.ar = ar
        self.ma = ma
        nlags, nvarall, nvars = ar.shape
        self.nlags, self.nvarall, self.nvars = (nlags, nvarall, nvars)
        self.isstructured = not (ar[0, :nvars] == np.eye(nvars)).all()
        if self.ma is None:
            self.ma = np.eye(nvars)[None, ...]
            self.isindependent = True
        else:
            self.isindependent = not (ma[0] == np.eye(nvars)).all()
        self.malags = ar.shape[0]
        self.hasexog = nvarall > nvars
        self.arm1 = -ar[1:]

    def vstack(self, a=None, name='ar'):
        """stack lagpolynomial vertically in 2d array

        """
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        return a.reshape(-1, self.nvarall)

    def hstack(self, a=None, name='ar'):
        """stack lagpolynomial horizontally in 2d array

        """
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        return a.swapaxes(1, 2).reshape(-1, self.nvarall).T

    def stacksquare(self, a=None, name='ar', orientation='vertical'):
        """stack lagpolynomial vertically in 2d square array with eye

        """
        if a is not None:
            a = a
        elif name == 'ar':
            a = self.ar
        elif name == 'ma':
            a = self.ma
        else:
            raise ValueError('no array or name given')
        astacked = a.reshape(-1, self.nvarall)
        lenpk, nvars = astacked.shape
        amat = np.eye(lenpk, k=nvars)
        amat[:, :nvars] = astacked
        return amat

    def vstackarma_minus1(self):
        """stack ar and lagpolynomial vertically in 2d array

        """
        a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
        return a.reshape(-1, self.nvarall)

    def hstackarma_minus1(self):
        """stack ar and lagpolynomial vertically in 2d array

        this is the Kalman Filter representation, I think
        """
        a = np.concatenate((self.ar[1:], self.ma[1:]), 0)
        return a.swapaxes(1, 2).reshape(-1, self.nvarall)

    def getisstationary(self, a=None):
        """check whether the auto-regressive lag-polynomial is stationary

        Returns
        -------
        isstationary : bool

        *attaches*

        areigenvalues : complex array
            eigenvalues sorted by absolute value

        References
        ----------
        formula taken from NAG manual

        """
        if a is not None:
            a = a
        elif self.isstructured:
            a = -self.reduceform(self.ar)[1:]
        else:
            a = -self.ar[1:]
        amat = self.stacksquare(a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.areigenvalues = ev
        return (np.abs(ev) < 1).all()

    def getisinvertible(self, a=None):
        """check whether the auto-regressive lag-polynomial is stationary

        Returns
        -------
        isinvertible : bool

        *attaches*

        maeigenvalues : complex array
            eigenvalues sorted by absolute value

        References
        ----------
        formula taken from NAG manual

        """
        if a is not None:
            a = a
        elif self.isindependent:
            a = self.reduceform(self.ma)[1:]
        else:
            a = self.ma[1:]
        if a.shape[0] == 0:
            self.maeigenvalues = np.array([], np.complex)
            return True
        amat = self.stacksquare(a)
        ev = np.sort(np.linalg.eigvals(amat))[::-1]
        self.maeigenvalues = ev
        return (np.abs(ev) < 1).all()

    def reduceform(self, apoly):
        """

        this assumes no exog, todo

        """
        if apoly.ndim != 3:
            raise ValueError('apoly needs to be 3d')
        nlags, nvarsex, nvars = apoly.shape
        a = np.empty_like(apoly)
        try:
            a0inv = np.linalg.inv(a[0, :nvars, :])
        except np.linalg.LinAlgError:
            raise ValueError('matrix not invertible', 'ask for implementation of pinv')
        for lag in range(nlags):
            a[lag] = np.dot(a0inv, apoly[lag])
        return a