import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class WordVectorsAccumulator(UsesDictionary):
    """Accumulate context vectors for words using word vector embeddings.

    Attributes
    ----------
    model: Word2Vec (:class:`~gensim.models.keyedvectors.KeyedVectors`)
        If None, a new Word2Vec model is trained on the given text corpus. Otherwise,
        it should be a pre-trained Word2Vec context vectors.
    model_kwargs:
        if model is None, these keyword arguments will be passed through to the Word2Vec constructor.
    """

    def __init__(self, relevant_ids, dictionary, model=None, **model_kwargs):
        super(WordVectorsAccumulator, self).__init__(relevant_ids, dictionary)
        self.model = model
        self.model_kwargs = model_kwargs

    def not_in_vocab(self, words):
        uniq_words = set(utils.flatten(words))
        return set((word for word in uniq_words if word not in self.model))

    def get_occurrences(self, word):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        try:
            self.token2id[word]
        except KeyError:
            word = self.dictionary.id2token[word]
        return self.model.get_vecattr(word, 'count')

    def get_co_occurrences(self, word1, word2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        raise NotImplementedError('Word2Vec model does not support co-occurrence counting')

    def accumulate(self, texts, window_size):
        if self.model is not None:
            logger.debug('model is already trained; no accumulation necessary')
            return self
        kwargs = self.model_kwargs.copy()
        if window_size is not None:
            kwargs['window'] = window_size
        kwargs['min_count'] = kwargs.get('min_count', 1)
        kwargs['sg'] = kwargs.get('sg', 1)
        kwargs['hs'] = kwargs.get('hw', 0)
        self.model = Word2Vec(**kwargs)
        self.model.build_vocab(texts)
        self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        self.model = self.model.wv
        return self

    def ids_similarity(self, ids1, ids2):
        words1 = self._words_with_embeddings(ids1)
        words2 = self._words_with_embeddings(ids2)
        return self.model.n_similarity(words1, words2)

    def _words_with_embeddings(self, ids):
        if not hasattr(ids, '__iter__'):
            ids = [ids]
        words = [self.dictionary.id2token[word_id] for word_id in ids]
        return [word for word in words if word in self.model]