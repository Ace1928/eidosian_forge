from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def _guess_n_features(self, corpus):
    """Attempt to guess number of features in `corpus`."""
    n_features = None
    if hasattr(corpus, 'dim'):
        n_features = corpus.dim
    elif hasattr(corpus, 'dictionary'):
        n_features = len(corpus.dictionary)
    elif hasattr(corpus, 'n_out'):
        n_features = corpus.n_out
    elif hasattr(corpus, 'num_terms'):
        n_features = corpus.num_terms
    elif isinstance(corpus, TransformedCorpus):
        try:
            return self._guess_n_features(corpus.obj)
        except TypeError:
            return self._guess_n_features(corpus.corpus)
    else:
        if not self.dim:
            raise TypeError("Couldn't find number of features, refusing to guess. Dimension: %s, corpus: %s)" % (self.dim, type(corpus)))
        logger.warning("Couldn't find number of features, trusting supplied dimension (%d)", self.dim)
        n_features = self.dim
    if self.dim and n_features != self.dim:
        logger.warning('Discovered inconsistent dataset dim (%d) and feature count from corpus (%d). Coercing to dimension given by argument.', self.dim, n_features)
    return n_features