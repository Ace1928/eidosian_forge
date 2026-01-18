from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
class UniformTermSimilarityIndex(TermSimilarityIndex):
    """
    Retrieves most similar terms for a given term under the hypothesis that the similarities between
    distinct terms are uniform.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
    term_similarity : float, optional
        The uniform similarity between distinct terms.

    See Also
    --------
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        A sparse term similarity matrix built using a term similarity index.

    Notes
    -----
    This class is mainly intended for testing SparseTermSimilarityMatrix and other classes that
    depend on the TermSimilarityIndex.

    """

    def __init__(self, dictionary, term_similarity=0.5):
        self.dictionary = sorted(dictionary.items())
        self.term_similarity = term_similarity

    def most_similar(self, t1, topn=10):
        for __, (t2_index, t2) in zip(range(topn), ((t2_index, t2) for t2_index, t2 in self.dictionary if t2 != t1)):
            yield (t2, self.term_similarity)