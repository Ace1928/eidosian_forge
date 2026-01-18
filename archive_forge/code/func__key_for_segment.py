import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def _key_for_segment(segment, topic_words):
    """A segment may have a single number of an iterable of them."""
    segment_key = tuple(segment) if hasattr(segment, '__iter__') else segment
    return (segment_key, topic_words)