import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
def _h_inverse_multiquadric(self, r):
    return 1.0 / np.sqrt((1.0 / self.epsilon * r) ** 2 + 1)