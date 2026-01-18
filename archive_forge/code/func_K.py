import numpy as np
from scipy.optimize import minimize
import GPy
from GPy.kern import Kern
from GPy.core import Param
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
def K(self, X, X2):
    if self.epsilon > 0.5:
        self.epsilon = 0.5
    if X2 is None:
        X2 = np.copy(X)
    T1 = X[:, 0].reshape(-1, 1)
    T2 = X2[:, 0].reshape(-1, 1)
    dists = pairwise_distances(T1, T2, 'cityblock')
    timekernel = (1 - self.epsilon) ** (0.5 * dists)
    X = X[:, 1:]
    X2 = X2[:, 1:]
    RBF = self.variance * np.exp(-np.square(euclidean_distances(X, X2)) / self.lengthscale)
    return RBF * timekernel