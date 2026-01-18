import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class WordOccurrenceAccumulator(WindowedTextsAnalyzer):
    """Accumulate word occurrences and co-occurrences from a sequence of corpus texts."""

    def __init__(self, *args):
        super(WordOccurrenceAccumulator, self).__init__(*args)
        self._occurrences = np.zeros(self._vocab_size, dtype='uint32')
        self._co_occurrences = sps.lil_matrix((self._vocab_size, self._vocab_size), dtype='uint32')
        self._uniq_words = np.zeros((self._vocab_size + 1,), dtype=bool)
        self._counter = Counter()

    def __str__(self):
        return self.__class__.__name__

    def accumulate(self, texts, window_size):
        self._co_occurrences = self._co_occurrences.tolil()
        self.partial_accumulate(texts, window_size)
        self._symmetrize()
        return self

    def partial_accumulate(self, texts, window_size):
        """Meant to be called several times to accumulate partial results.

        Notes
        -----
        The final accumulation should be performed with the `accumulate` method as opposed to this one.
        This method does not ensure the co-occurrence matrix is in lil format and does not
        symmetrize it after accumulation.

        """
        self._current_doc_num = -1
        self._token_at_edge = None
        self._counter.clear()
        super(WordOccurrenceAccumulator, self).accumulate(texts, window_size)
        for combo, count in self._counter.items():
            self._co_occurrences[combo] += count
        return self

    def analyze_text(self, window, doc_num=None):
        self._slide_window(window, doc_num)
        mask = self._uniq_words[:-1]
        if mask.any():
            self._occurrences[mask] += 1
            self._counter.update(itertools.combinations(np.nonzero(mask)[0], 2))

    def _slide_window(self, window, doc_num):
        if doc_num != self._current_doc_num:
            self._uniq_words[:] = False
            self._uniq_words[np.unique(window)] = True
            self._current_doc_num = doc_num
        else:
            self._uniq_words[self._token_at_edge] = False
            self._uniq_words[window[-1]] = True
        self._token_at_edge = window[0]

    def _symmetrize(self):
        """Word pairs may have been encountered in (i, j) and (j, i) order.

        Notes
        -----
        Rather than enforcing a particular ordering during the update process,
        we choose to symmetrize the co-occurrence matrix after accumulation has completed.

        """
        co_occ = self._co_occurrences
        co_occ.setdiag(self._occurrences)
        self._co_occurrences = co_occ + co_occ.T - sps.diags(co_occ.diagonal(), offsets=0, dtype='uint32')

    def _get_occurrences(self, word_id):
        return self._occurrences[word_id]

    def _get_co_occurrences(self, word_id1, word_id2):
        return self._co_occurrences[word_id1, word_id2]

    def merge(self, other):
        self._occurrences += other._occurrences
        self._co_occurrences += other._co_occurrences
        self._num_docs += other._num_docs