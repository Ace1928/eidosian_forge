import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
@staticmethod
def _clip_vectors(vectors, epsilon):
    """Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D, or 2-D (in which case the norm for each row is checked).
        epsilon : float
            Parameter for numerical stability, each dimension of the vector is reduced by `epsilon`
            if the norm of the vector is greater than or equal to 1.

        Returns
        -------
        numpy.array
            Array with norms clipped below 1.

        """
    one_d = len(vectors.shape) == 1
    threshold = 1 - epsilon
    if one_d:
        norm = np.linalg.norm(vectors)
        if norm < threshold:
            return vectors
        else:
            return vectors / norm - np.sign(vectors) * epsilon
    else:
        norms = np.linalg.norm(vectors, axis=1)
        if (norms < threshold).all():
            return vectors
        else:
            vectors[norms >= threshold] *= (threshold / norms[norms >= threshold])[:, np.newaxis]
            vectors[norms >= threshold] -= np.sign(vectors[norms >= threshold]) * epsilon
            return vectors