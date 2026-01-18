import numpy as np
import matplotlib.pyplot as plt
class JumpDiffusionKou:

    def __init__(self):
        pass

    def simulate(self, m, s, lambd, p, e1, e2, ts, nrepl):
        T = ts[-1]
        N = np.random.poisson(lambd * T, size=(nrepl, 1))
        jumps = []
        nobs = len(ts)
        jumps = np.zeros((nrepl, nobs))
        for j in range(nrepl):
            t = T * np.random.rand(N[j])
            t = np.sort(t)
            ww = np.random.binomial(1, p, size=N[j])
            S = ww * np.random.exponential(e1, size=N[j]) - (1 - ww) * np.random.exponential(e2, N[j])
            CumS = np.cumsum(S)
            jumps_ts = np.zeros(nobs)
            for n in range(nobs):
                Events = sum(t <= ts[n]) - 1
                jumps_ts[n] = 0
                if Events:
                    jumps_ts[n] = CumS[Events]
            jumps[j, :] = jumps_ts
        D_Diff = np.zeros((nrepl, nobs))
        for k in range(nobs):
            Dt = ts[k]
            if k > 1:
                Dt = ts[k] - ts[k - 1]
            D_Diff[:, k] = m * Dt + s * np.sqrt(Dt) * np.random.normal(size=nrepl)
        x = np.hstack((np.zeros((nrepl, 1)), np.cumsum(D_Diff, 1) + jumps))
        return x