from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import math
from gensim import interfaces, utils
import numpy as np
class BM25ABC(interfaces.TransformationABC, metaclass=ABCMeta):
    """Objects of this abstract class realize the transformation between word-document co-occurrence
    matrix (int) into a BM25 matrix (positive floats). Concrete subclasses of this abstract class
    implement different BM25 scoring functions.

    """

    def __init__(self, corpus=None, dictionary=None):
        """Pre-compute the average length of a document and inverse term document frequencies,
        which will be used to weight term frequencies for the documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int) or None, optional
            An input corpus, which will be used to compute the average length of a document and
            inverse term document frequencies. If None, then `dictionary` will be used to compute
            the statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        dictionary : :class:`~gensim.corpora.Dictionary`
            An input dictionary, which will be used to compute the average length of a document and
            inverse term document frequencies.  If None, then `corpus` will be used to compute the
            statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.

        Attributes
        ----------
        avgdl : float
            The average length of a document.
        idfs : dict of (int, float)
            A mapping from term ids to inverse term document frequencies.

        """
        self.avgdl, self.idfs = (None, None)
        if dictionary:
            if corpus:
                logger.warning('constructor received both corpus and dictionary; ignoring the corpus')
            num_tokens = sum(dictionary.cfs.values())
            self.avgdl = num_tokens / dictionary.num_docs
            self.idfs = self.precompute_idfs(dictionary.dfs, dictionary.num_docs)
        elif corpus:
            dfs = defaultdict(lambda: 0)
            num_tokens = 0
            num_docs = 0
            for bow in corpus:
                num_tokens += len(bow)
                for term_id in set((term_id for term_id, _ in bow)):
                    dfs[term_id] += 1
                num_docs += 1
            self.avgdl = num_tokens / num_docs
            self.idfs = self.precompute_idfs(dfs, num_docs)
        else:
            pass

    @abstractmethod
    def precompute_idfs(self, dfs, num_docs):
        """Precompute inverse term document frequencies, which will be used to weight term frequencies
        for the documents.

        Parameters
        ----------
        dfs : dict of (int, int)
            A mapping from term ids to term document frequencies.
        num_docs : int
            The total number of documents in the training corpus.

        Returns
        -------
        idfs : dict of (int, float)
            A mapping from term ids to inverse term document frequencies.

        """
        pass

    @abstractmethod
    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        """Compute vector space weights for a set of terms in a document.

        Parameters
        ----------
        num_tokens : int
            The number of tokens in the document.
        term_frequencies : ndarray
            1D array of term frequencies.
        idfs : ndarray
            1D array of inverse term document frequencies.

        Returns
        -------
        term_weights : ndarray
            1D array of vector space weights.

        """
        pass

    def __getitem__(self, bow):
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)
        num_tokens = sum((freq for term_id, freq in bow))
        term_ids, term_frequencies, idfs = ([], [], [])
        for term_id, term_frequency in bow:
            term_ids.append(term_id)
            term_frequencies.append(term_frequency)
            idfs.append(self.idfs.get(term_id) or 0.0)
        term_frequencies, idfs = (np.array(term_frequencies), np.array(idfs))
        term_weights = self.get_term_weights(num_tokens, term_frequencies, idfs)
        vector = [(term_id, float(weight)) for term_id, weight in zip(term_ids, term_weights)]
        return vector