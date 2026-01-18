from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
class TermSimilarityIndex(SaveLoad):
    """
    Base class = common interface for retrieving the most similar terms for a given term.

    See Also
    --------
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        A sparse term similarity matrix built using a term similarity index.

    """

    def most_similar(self, term, topn=10):
        """Get most similar terms for a given term.

        Return the most similar terms for a given term along with their similarities.

        Parameters
        ----------
        term : str
            The term for which we are retrieving `topn` most similar terms.
        topn : int, optional
            The maximum number of most similar terms to `term` that will be retrieved.

        Returns
        -------
        iterable of (str, float)
            Most similar terms along with their similarities to `term`. Only terms distinct from
            `term` must be returned.

        """
        raise NotImplementedError

    def __str__(self):
        members = ', '.join(('%s=%s' % pair for pair in vars(self).items()))
        return '%s<%s>' % (self.__class__.__name__, members)