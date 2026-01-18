import copy
import random
import sys
from nltk.cluster.util import VectorSpaceClusterer
def _sum_distances(self, vectors1, vectors2):
    difference = 0.0
    for u, v in zip(vectors1, vectors2):
        difference += self._distance(u, v)
    return difference