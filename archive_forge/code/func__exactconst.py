import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def _exactconst(self, expnt):
    return (1 - expnt) * (self.mu - self.sigma ** 2 / 2.0 / self.kappa)