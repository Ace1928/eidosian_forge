from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
class AutoDampedGradientDescentSolver(GradientDescentSolver):

    def __init__(self, tgt_oscill=0.1, start_damp=0.1, nhistory=4, tgt_pow=0.3):
        self.tgt_oscill = tgt_oscill
        self.cur_damp = start_damp
        self.nhistory = nhistory
        self.tgt_pow = tgt_pow
        self.history_damping = []

    def damping(self, iter_idx, mx_iter):
        if iter_idx >= self.nhistory:
            hist = self.history_rms_f[-self.nhistory:]
            avg = np.mean(hist)
            even = hist[::2]
            odd = hist[1::2]
            signed_metric = sum(even - avg) - sum(odd - avg)
            oscillatory_metric = abs(signed_metric) / (avg * self.nhistory)
            self.cur_damp *= (self.tgt_oscill / oscillatory_metric) ** self.tgt_pow
        self.history_damping.append(self.cur_damp)
        return self.cur_damp