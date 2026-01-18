from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
class PolakRibiereConjugateGradientSolver(SolverBase):

    def __init__(self, reset_freq=10):
        self.reset_freq = reset_freq

    def alloc(self):
        self.history_a = []
        self.history_Bn = []
        self.history_sn = []
        super(PolakRibiereConjugateGradientSolver, self).alloc()

    def step(self, x, iter_idx, maxiter):
        dx = self.history_dx
        sn = self.history_sn
        if iter_idx in (0, 1) or iter_idx % self.reset_freq == 0:
            dxn = self.line_search(x, self._gd_step(x))
            sn.append(x * 0)
        else:
            dx0 = dx[-1]
            dx1 = dx[-2]
            ddx01 = dx0 - dx1
            Bn = dx0.dot(ddx01) / dx1.dot(dx1)
            self.history_Bn.append(Bn)
            sn.append(dx[-1] + Bn * sn[-1])
            a = self.line_search(x, sn[-1])
            self.history_a.append(a)
            dxn = a * sn[-1]
        return dxn