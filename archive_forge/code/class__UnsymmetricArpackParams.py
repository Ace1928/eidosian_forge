import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import isdense, is_pydata_spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock
from . import _arpack
class _UnsymmetricArpackParams(_ArpackParams):

    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None, Minv_matvec=None, sigma=None, ncv=None, v0=None, maxiter=None, which='LM', tol=0):
        if mode == 1:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=1')
            if M_matvec is not None:
                raise ValueError('M_matvec cannot be specified for mode=1')
            if Minv_matvec is not None:
                raise ValueError('Minv_matvec cannot be specified for mode=1')
            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError('matvec must be specified for mode=2')
            if M_matvec is None:
                raise ValueError('M_matvec must be specified for mode=2')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode=2')
            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode in (3, 4):
            if matvec is None:
                raise ValueError('matvec must be specified for mode in (3,4)')
            if Minv_matvec is None:
                raise ValueError('Minv_matvec must be specified for mode in (3,4)')
            self.matvec = matvec
            if tp in 'DF':
                if mode == 3:
                    self.OPa = Minv_matvec
                else:
                    raise ValueError('mode=4 invalid for complex A')
            elif mode == 3:
                self.OPa = lambda x: np.real(Minv_matvec(x))
            else:
                self.OPa = lambda x: np.imag(Minv_matvec(x))
            if M_matvec is None:
                self.B = lambda x: x
                self.bmat = 'I'
                self.OP = self.OPa
            else:
                self.B = M_matvec
                self.bmat = 'G'
                self.OP = lambda x: self.OPa(M_matvec(x))
        else:
            raise ValueError('mode=%i not implemented' % mode)
        if which not in _NEUPD_WHICH:
            raise ValueError('Parameter which must be one of %s' % ' '.join(_NEUPD_WHICH))
        if k >= n - 1:
            raise ValueError('k must be less than ndim(A)-1, k=%d' % k)
        _ArpackParams.__init__(self, n, k, tp, mode, sigma, ncv, v0, maxiter, which, tol)
        if self.ncv > n or self.ncv <= k + 1:
            raise ValueError('ncv must be k+1<ncv<=n, ncv=%s' % self.ncv)
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(3 * self.ncv * (self.ncv + 2), self.tp)
        ltr = _type_conv[self.tp]
        self._arpack_solver = _arpack.__dict__[ltr + 'naupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'neupd']
        self.iterate_infodict = _NAUPD_ERRORS[ltr]
        self.extract_infodict = _NEUPD_ERRORS[ltr]
        self.ipntr = np.zeros(14, arpack_int)
        if self.tp in 'FD':
            self.rwork = _aligned_zeros(self.ncv, self.tp.lower())
        else:
            self.rwork = None

    def iterate(self):
        if self.tp in 'fd':
            results = self._arpack_solver(self.ido, self.bmat, self.which, self.k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.info)
            self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info = results
        else:
            results = self._arpack_solver(self.ido, self.bmat, self.which, self.k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.rwork, self.info)
            self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info = results
        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            if self.mode in (1, 2):
                self.workd[yslice] = self.OP(self.workd[xslice])
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError('ARPACK requested user shifts.  Assure ISHIFT==0')
        else:
            self.converged = True
            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        k, n = (self.k, self.n)
        ierr = 0
        howmny = 'A'
        sselect = np.zeros(self.ncv, 'int')
        sigmar = np.real(self.sigma)
        sigmai = np.imag(self.sigma)
        workev = np.zeros(3 * self.ncv, self.tp)
        if self.tp in 'fd':
            dr = np.zeros(k + 1, self.tp)
            di = np.zeros(k + 1, self.tp)
            zr = np.zeros((n, k + 1), self.tp)
            dr, di, zr, ierr = self._arpack_extract(return_eigenvectors, howmny, sselect, sigmar, sigmai, workev, self.bmat, self.which, k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.info)
            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)
            nreturned = self.iparam[4]
            d = dr + 1j * di
            z = zr.astype(self.tp.upper())
            if sigmai == 0:
                i = 0
                while i <= k:
                    if abs(d[i].imag) != 0:
                        if i < k:
                            z[:, i] = zr[:, i] + 1j * zr[:, i + 1]
                            z[:, i + 1] = z[:, i].conjugate()
                            i += 1
                        else:
                            nreturned -= 1
                    i += 1
            else:
                i = 0
                while i <= k:
                    if abs(d[i].imag) == 0:
                        d[i] = np.dot(zr[:, i], self.matvec(zr[:, i]))
                    elif i < k:
                        z[:, i] = zr[:, i] + 1j * zr[:, i + 1]
                        z[:, i + 1] = z[:, i].conjugate()
                        d[i] = np.dot(zr[:, i], self.matvec(zr[:, i])) + np.dot(zr[:, i + 1], self.matvec(zr[:, i + 1])) + 1j * (np.dot(zr[:, i], self.matvec(zr[:, i + 1])) - np.dot(zr[:, i + 1], self.matvec(zr[:, i])))
                        d[i + 1] = d[i].conj()
                        i += 1
                    else:
                        nreturned -= 1
                    i += 1
            if nreturned <= k:
                d = d[:nreturned]
                z = z[:, :nreturned]
            else:
                if self.mode in (1, 2):
                    rd = d
                elif self.mode in (3, 4):
                    rd = 1 / (d - self.sigma)
                if self.which in ['LR', 'SR']:
                    ind = np.argsort(rd.real)
                elif self.which in ['LI', 'SI']:
                    ind = np.argsort(abs(rd.imag))
                else:
                    ind = np.argsort(abs(rd))
                if self.which in ['LR', 'LM', 'LI']:
                    ind = ind[-k:][::-1]
                elif self.which in ['SR', 'SM', 'SI']:
                    ind = ind[:k]
                d = d[ind]
                z = z[:, ind]
        else:
            d, z, ierr = self._arpack_extract(return_eigenvectors, howmny, sselect, self.sigma, workev, self.bmat, self.which, k, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.workd, self.workl, self.rwork, ierr)
            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)
            k_ok = self.iparam[4]
            d = d[:k_ok]
            z = z[:, :k_ok]
        if return_eigenvectors:
            return (d, z)
        else:
            return d