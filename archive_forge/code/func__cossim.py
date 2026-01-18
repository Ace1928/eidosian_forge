import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def _cossim(cv1, cv2):
    return cv1.T.dot(cv2)[0, 0] / (_magnitude(cv1) * _magnitude(cv2))