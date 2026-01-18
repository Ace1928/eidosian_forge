import numpy as np
class LeadSelfEnergy:
    conv = 1e-08

    def __init__(self, hs_dii, hs_dij, hs_dim, eta=0.0001):
        self.h_ii, self.s_ii = hs_dii
        self.h_ij, self.s_ij = hs_dij
        self.h_im, self.s_im = hs_dim
        self.nbf = self.h_im.shape[1]
        self.eta = eta
        self.energy = None
        self.bias = 0
        self.sigma_mm = np.empty((self.nbf, self.nbf), complex)

    def retarded(self, energy):
        """Return self-energy (sigma) evaluated at specified energy."""
        if energy != self.energy:
            self.energy = energy
            z = energy - self.bias + self.eta * 1j
            tau_im = z * self.s_im - self.h_im
            a_im = np.linalg.solve(self.get_sgfinv(energy), tau_im)
            tau_mi = z * self.s_im.T.conj() - self.h_im.T.conj()
            self.sigma_mm[:] = np.dot(tau_mi, a_im)
        return self.sigma_mm

    def set_bias(self, bias):
        self.bias = bias

    def get_lambda(self, energy):
        """Return the lambda (aka Gamma) defined by i(S-S^d).

        Here S is the retarded selfenergy, and d denotes the hermitian
        conjugate.
        """
        sigma_mm = self.retarded(energy)
        return 1j * (sigma_mm - sigma_mm.T.conj())

    def get_sgfinv(self, energy):
        """The inverse of the retarded surface Green function"""
        z = energy - self.bias + self.eta * 1j
        v_00 = z * self.s_ii.T.conj() - self.h_ii.T.conj()
        v_11 = v_00.copy()
        v_10 = z * self.s_ij - self.h_ij
        v_01 = z * self.s_ij.T.conj() - self.h_ij.T.conj()
        delta = self.conv + 1
        while delta > self.conv:
            a = np.linalg.solve(v_11, v_01)
            b = np.linalg.solve(v_11, v_10)
            v_01_dot_b = np.dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= np.dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -np.dot(v_01, a)
            v_10 = -np.dot(v_10, b)
            delta = abs(v_01).max()
        return v_00