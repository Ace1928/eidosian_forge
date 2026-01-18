import numpy as np
import matplotlib.pyplot as plt
class IG:
    """inverse-Gaussian ??? used by NIG
    """

    def __init__(self):
        pass

    def simulate(self, l, m, nrepl):
        N = np.random.randn(nrepl, 1)
        Y = N ** 2
        X = m + 0.5 * m * m / l * Y - 0.5 * m / l * np.sqrt(4 * m * l * Y + m * m * Y ** 2)
        U = np.random.rand(nrepl, 1)
        ind = U > m / (X + m)
        X[ind] = m * m / X[ind]
        return X.ravel()