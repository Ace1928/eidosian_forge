import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
def _h_thin_plate(self, r):
    return xlogy(r ** 2, r)