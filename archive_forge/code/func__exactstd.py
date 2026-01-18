import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def _exactstd(self, expnt):
    return self.sigma * np.sqrt((1 - expnt * expnt) / 2.0 / self.kappa)