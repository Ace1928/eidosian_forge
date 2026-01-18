import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def _magnitude(sparse_vec):
    return np.sqrt(np.sum(sparse_vec.data ** 2))