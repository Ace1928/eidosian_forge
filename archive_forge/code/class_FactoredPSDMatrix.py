import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
class FactoredPSDMatrix:
    """
    Representation of a positive semidefinite matrix in factored form.

    The representation is constructed based on a vector `diag` and
    rectangular matrix `root`, such that the PSD matrix represented by
    the class instance is Diag + root * root', where Diag is the
    square diagonal matrix with `diag` on its main diagonal.

    Parameters
    ----------
    diag : 1d array_like
        See above
    root : 2d array_like
        See above

    Notes
    -----
    The matrix is represented internally in the form Diag^{1/2}(I +
    factor * scales * factor')Diag^{1/2}, where `Diag` and `scales`
    are diagonal matrices, and `factor` is an orthogonal matrix.
    """

    def __init__(self, diag, root):
        self.diag = diag
        self.root = root
        root = root / np.sqrt(diag)[:, None]
        u, s, vt = np.linalg.svd(root, 0)
        self.factor = u
        self.scales = s ** 2

    def to_matrix(self):
        """
        Returns the PSD matrix represented by this instance as a full
        (square) matrix.
        """
        return np.diag(self.diag) + np.dot(self.root, self.root.T)

    def decorrelate(self, rhs):
        """
        Decorrelate the columns of `rhs`.

        Parameters
        ----------
        rhs : array_like
            A 2 dimensional array with the same number of rows as the
            PSD matrix represented by the class instance.

        Returns
        -------
        C^{-1/2} * rhs, where C is the covariance matrix represented
        by this class instance.

        Notes
        -----
        The returned matrix has the identity matrix as its row-wise
        population covariance matrix.

        This function exploits the factor structure for efficiency.
        """
        qval = -1 + 1 / np.sqrt(1 + self.scales)
        rhs = rhs / np.sqrt(self.diag)[:, None]
        rhs1 = np.dot(self.factor.T, rhs)
        rhs1 *= qval[:, None]
        rhs1 = np.dot(self.factor, rhs1)
        rhs += rhs1
        return rhs

    def solve(self, rhs):
        """
        Solve a linear system of equations with factor-structured
        coefficients.

        Parameters
        ----------
        rhs : array_like
            A 2 dimensional array with the same number of rows as the
            PSD matrix represented by the class instance.

        Returns
        -------
        C^{-1} * rhs, where C is the covariance matrix represented
        by this class instance.

        Notes
        -----
        This function exploits the factor structure for efficiency.
        """
        qval = -self.scales / (1 + self.scales)
        dr = np.sqrt(self.diag)
        rhs = rhs / dr[:, None]
        mat = qval[:, None] * np.dot(self.factor.T, rhs)
        rhs = rhs + np.dot(self.factor, mat)
        return rhs / dr[:, None]

    def logdet(self):
        """
        Returns the logarithm of the determinant of a
        factor-structured matrix.
        """
        logdet = np.sum(np.log(self.diag))
        logdet += np.sum(np.log(self.scales))
        logdet += np.sum(np.log(1 + 1 / self.scales))
        return logdet