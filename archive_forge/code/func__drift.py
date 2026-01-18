import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def _drift(self, *args, **kwds):
    x = kwds['x']
    return self.lambd * (self.mu - x)