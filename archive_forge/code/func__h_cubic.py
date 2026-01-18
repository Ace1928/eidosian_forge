import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
def _h_cubic(self, r):
    return r ** 3