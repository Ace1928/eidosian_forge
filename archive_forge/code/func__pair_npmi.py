import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def _pair_npmi(pair, accumulator):
    """Compute normalized pairwise mutual information (**NPMI**) between a pair of words.

    Parameters
    ----------
    pair : (int, int)
        The pair of words (word_id1, word_id2).
    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from probability_estimation.

    Return
    ------
    float
        NPMI between a pair of words.

    """
    return log_ratio_measure([[pair]], accumulator, True)[0]