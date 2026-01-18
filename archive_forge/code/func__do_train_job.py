import logging
import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer
from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
def _do_train_job(self, job, alpha, inits):
    """Train model using `job` data.

        Parameters
        ----------
        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            The corpus chunk to be used for training this batch.
        alpha : float
            Learning rate to be used for training this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
    work, neu1 = inits
    tally = 0
    for doc in job:
        doctag_indexes = [self.dv.get_index(tag) for tag in doc.tags if tag in self.dv]
        doctag_vectors = self.dv.vectors
        doctags_lockf = self.dv.vectors_lockf
        if self.sg:
            tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work, train_words=self.dbow_words, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
        elif self.dm_concat:
            tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
        else:
            tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
    return (tally, self._raw_word_count(job))