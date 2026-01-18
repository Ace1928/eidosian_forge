from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
class WordEmbeddingSimilarityIndex(TermSimilarityIndex):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar terms for a given term.

    Notes
    -----
    By fitting the word embeddings to a vocabulary that you will be using, you
    can eliminate all out-of-vocabulary (OOV) words that you would otherwise
    receive from the `most_similar` method. In subword models such as fastText,
    this procedure will also infer word-vectors for words from your vocabulary
    that previously had no word-vector.

    >>> from gensim.test.utils import common_texts, datapath
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import FastText
    >>> from gensim.models.word2vec import LineSentence
    >>> from gensim.similarities import WordEmbeddingSimilarityIndex
    >>>
    >>> model = FastText(common_texts, vector_size=20, min_count=1)  # train word-vectors on a corpus
    >>> different_corpus = LineSentence(datapath('lee_background.cor'))
    >>> dictionary = Dictionary(different_corpus)  # construct a vocabulary on a different corpus
    >>> words = [word for word, count in dictionary.most_common()]
    >>> word_vectors = model.wv.vectors_for_all(words)  # remove OOV word-vectors and infer word-vectors for new words
    >>> assert len(dictionary) == len(word_vectors)  # all words from our vocabulary received their word-vectors
    >>> termsim_index = WordEmbeddingSimilarityIndex(word_vectors)

    Parameters
    ----------
    keyedvectors : :class:`~gensim.models.keyedvectors.KeyedVectors`
        The word embeddings.
    threshold : float, optional
        Only embeddings more similar than `threshold` are considered when retrieving word embeddings
        closest to a given word embedding.
    exponent : float, optional
        Take the word embedding similarities larger than `threshold` to the power of `exponent`.
    kwargs : dict or None
        A dict with keyword arguments that will be passed to the
        :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar` method
        when retrieving the word embeddings closest to a given word embedding.

    See Also
    --------
    :class:`~gensim.similarities.levenshtein.LevenshteinSimilarityIndex`
        Retrieve most similar terms for a given term using the Levenshtein distance.
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    """

    def __init__(self, keyedvectors, threshold=0.0, exponent=2.0, kwargs=None):
        self.keyedvectors = keyedvectors
        self.threshold = threshold
        self.exponent = exponent
        self.kwargs = kwargs or {}
        super(WordEmbeddingSimilarityIndex, self).__init__()

    def most_similar(self, t1, topn=10):
        if t1 not in self.keyedvectors:
            logger.debug('an out-of-dictionary term "%s"', t1)
        else:
            most_similar = self.keyedvectors.most_similar(positive=[t1], topn=topn, **self.kwargs)
            for t2, similarity in most_similar:
                if similarity > self.threshold:
                    yield (t2, similarity ** self.exponent)