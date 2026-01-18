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
def _handle_duplicates(vector_updates, node_indices):
    """Handle occurrences of multiple updates to the same node in a batch of vector updates.

        Parameters
        ----------
        vector_updates : numpy.array
            Array with each row containing updates to be performed on a certain node.
        node_indices : list of int
            Node indices on which the above updates are to be performed on.

        Notes
        -----
        Mutates the `vector_updates` array.

        Required because vectors[[2, 1, 2]] += np.array([-0.5, 1.0, 0.5]) performs only the last update
        on the row at index 2.

        """
    counts = Counter(node_indices)
    node_dict = defaultdict(list)
    for i, node_index in enumerate(node_indices):
        node_dict[node_index].append(i)
    for node_index, count in counts.items():
        if count == 1:
            continue
        positions = node_dict[node_index]
        vector_updates[positions[-1]] = vector_updates[positions].sum(axis=0)
        vector_updates[positions[:-1]] = 0