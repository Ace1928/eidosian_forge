import numpy as np
import matplotlib.pyplot as plt
class NIG:
    """normal-inverse-Gaussian
    """

    def __init__(self):
        pass

    def simulate(self, th, k, s, ts, nrepl):
        T = len(ts)
        DXs = np.zeros((nrepl, T))
        for t in range(T):
            Dt = ts[1] - 0
            if t > 1:
                Dt = ts[t] - ts[t - 1]
            lfrac = 1 / k * Dt ** 2
            m = Dt
            DS = IG().simulate(lfrac, m, nrepl)
            N = np.random.randn(nrepl)
            DX = s * N * np.sqrt(DS) + th * DS
            DXs[:, t] = DX
        x = np.cumsum(DXs, 1)
        return x