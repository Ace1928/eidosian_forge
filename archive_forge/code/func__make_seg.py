import itertools
import logging
import numpy as np
import scipy.sparse as sps
from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure
def _make_seg(self, segment_word_ids, topic_word_ids):
    """Return context vectors for segmentation (Internal helper function).

        Parameters
        ----------
        segment_word_ids : iterable or int
            Ids of words in segment.
        topic_word_ids : list
            Ids of words in topic.
        Returns
        -------
        csr_matrix :class:`~scipy.sparse.csr`
            Matrix in Compressed Sparse Row format

        """
    context_vector = sps.lil_matrix((self.vocab_size, 1))
    if not hasattr(segment_word_ids, '__iter__'):
        segment_word_ids = (segment_word_ids,)
    for w_j in topic_word_ids:
        idx = (self.mapping[w_j], 0)
        for pair in (tuple(sorted((w_i, w_j))) for w_i in segment_word_ids):
            if pair not in self.sim_cache:
                self.sim_cache[pair] = self.similarity(pair, self.accumulator)
            context_vector[idx] += self.sim_cache[pair] ** self.gamma
    return context_vector.tocsr()