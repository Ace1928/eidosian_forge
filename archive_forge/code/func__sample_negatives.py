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
def _sample_negatives(self, node_index):
    """Get a sample of negatives for the given node.

        Parameters
        ----------
        node_index : int
            Index of the positive node for which negative samples are to be returned.

        Returns
        -------
        numpy.array
            Array of shape (self.negative,) containing indices of negative nodes for the given node index.

        """
    node_relations = self.node_relations[node_index]
    num_remaining_nodes = len(self.kv) - len(node_relations)
    if num_remaining_nodes < self.negative:
        raise ValueError('Cannot sample %d negative nodes from a set of %d negative nodes for %s' % (self.negative, num_remaining_nodes, self.kv.index_to_key[node_index]))
    positive_fraction = float(len(node_relations)) / len(self.kv)
    if positive_fraction < 0.01:
        indices = self._get_candidate_negatives()
        unique_indices = set(indices)
        times_sampled = 1
        while len(indices) != len(unique_indices) or unique_indices & node_relations:
            times_sampled += 1
            indices = self._get_candidate_negatives()
            unique_indices = set(indices)
        if times_sampled > 1:
            logger.debug('sampled %d times, positive fraction %.5f', times_sampled, positive_fraction)
    else:
        valid_negatives = np.array(list(self.indices_set - node_relations))
        probs = self._node_probabilities[valid_negatives]
        probs /= probs.sum()
        indices = self._np_random.choice(valid_negatives, size=self.negative, p=probs, replace=False)
    return list(indices)