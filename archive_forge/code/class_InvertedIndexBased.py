import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class InvertedIndexBased(BaseAnalyzer):
    """Analyzer that builds up an inverted index to accumulate stats."""

    def __init__(self, *args):
        """

        Parameters
        ----------
        args : dict
            Look at :class:`~gensim.topic_coherence.text_analysis.BaseAnalyzer`

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.topic_coherence import text_analysis
            >>>
            >>> ids = {1: 'fake', 4: 'cats'}
            >>> ininb = text_analysis.InvertedIndexBased(ids)
            >>>
            >>> print(ininb._inverted_index)
            [set([]) set([])]

        """
        super(InvertedIndexBased, self).__init__(*args)
        self._inverted_index = np.array([set() for _ in range(self._vocab_size)])

    def _get_occurrences(self, word_id):
        return len(self._inverted_index[word_id])

    def _get_co_occurrences(self, word_id1, word_id2):
        s1 = self._inverted_index[word_id1]
        s2 = self._inverted_index[word_id2]
        return len(s1.intersection(s2))

    def index_to_dict(self):
        contiguous2id = {n: word_id for word_id, n in self.id2contiguous.items()}
        return {contiguous2id[n]: doc_id_set for n, doc_id_set in enumerate(self._inverted_index)}