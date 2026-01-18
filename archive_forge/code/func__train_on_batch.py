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
def _train_on_batch(self, relations, check_gradients=False):
    """Perform training for a single training batch.

        Parameters
        ----------
        relations : list of tuples of (int, int)
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        check_gradients : bool, optional
            Whether to compare the computed gradients to autograd gradients for this batch.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            The batch that was just trained on, contains computed loss for the batch.

        """
    all_negatives = self._sample_negatives_batch((relation[0] for relation in relations))
    batch = self._prepare_training_batch(relations, all_negatives, check_gradients)
    self._update_vectors_batch(batch)
    return batch