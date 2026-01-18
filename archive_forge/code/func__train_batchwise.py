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
def _train_batchwise(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
    """Train Poincare embeddings using specified parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.
        batch_size : int, optional
            Number of examples to train on in a single batch.
        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.
        check_gradients_every : int or None, optional
            Compares computed gradients and autograd gradients after every `check_gradients_every` batches.
            Useful for debugging, doesn't compare by default.

        """
    if self.workers > 1:
        raise NotImplementedError('Multi-threaded version not implemented yet')
    for epoch in range(1, epochs + 1):
        indices = list(range(len(self.all_relations)))
        self._np_random.shuffle(indices)
        avg_loss = 0.0
        last_time = time.time()
        for batch_num, i in enumerate(range(0, len(indices), batch_size), start=1):
            should_print = not batch_num % print_every
            check_gradients = bool(check_gradients_every) and batch_num % check_gradients_every == 0
            batch_indices = indices[i:i + batch_size]
            relations = [self.all_relations[idx] for idx in batch_indices]
            result = self._train_on_batch(relations, check_gradients=check_gradients)
            avg_loss += result.loss
            if should_print:
                avg_loss /= print_every
                time_taken = time.time() - last_time
                speed = print_every * batch_size / time_taken
                logger.info('training on epoch %d, examples #%d-#%d, loss: %.2f' % (epoch, i, i + batch_size, avg_loss))
                logger.info('time taken for %d examples: %.2f s, %.2f examples / s' % (print_every * batch_size, time_taken, speed))
                last_time = time.time()
                avg_loss = 0.0