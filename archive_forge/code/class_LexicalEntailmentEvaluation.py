import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
class LexicalEntailmentEvaluation:
    """Evaluate reconstruction on given network for any embedding."""

    def __init__(self, filepath):
        """Initialize evaluation instance with HyperLex text file containing relation pairs.

        Parameters
        ----------
        filepath : str
            Path to HyperLex text file.

        """
        expected_scores = {}
        with utils.open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=' ')
            for row in reader:
                word_1, word_2 = (row['WORD1'], row['WORD2'])
                expected_scores[word_1, word_2] = float(row['AVG_SCORE'])
        self.scores = expected_scores
        self.alpha = 1000

    def score_function(self, embedding, trie, term_1, term_2):
        """Compute predicted score - extent to which `term_1` is a type of `term_2`.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to use for computing predicted score.
        trie : :class:`pygtrie.Trie`
            Trie to use for finding matching vocab terms for input terms.
        term_1 : str
            Input term.
        term_2 : str
            Input term.

        Returns
        -------
        float
            Predicted score (the extent to which `term_1` is a type of `term_2`).

        """
        try:
            word_1_terms = self.find_matching_terms(trie, term_1)
            word_2_terms = self.find_matching_terms(trie, term_2)
        except KeyError:
            raise ValueError('No matching terms found for either %s or %s' % (term_1, term_2))
        min_distance = np.inf
        min_term_1, min_term_2 = (None, None)
        for term_1 in word_1_terms:
            for term_2 in word_2_terms:
                distance = embedding.distance(term_1, term_2)
                if distance < min_distance:
                    min_term_1, min_term_2 = (term_1, term_2)
                    min_distance = distance
        assert min_term_1 is not None and min_term_2 is not None
        vector_1, vector_2 = (embedding.get_vector(min_term_1), embedding.get_vector(min_term_2))
        norm_1, norm_2 = (np.linalg.norm(vector_1), np.linalg.norm(vector_2))
        return -1 * (1 + self.alpha * (norm_2 - norm_1)) * min_distance

    @staticmethod
    def find_matching_terms(trie, word):
        """Find terms in the `trie` beginning with the `word`.

        Parameters
        ----------
        trie : :class:`pygtrie.Trie`
            Trie to use for finding matching terms.
        word : str
            Input word to use for prefix search.

        Returns
        -------
        list of str
            List of matching terms.

        """
        matches = trie.items('%s.' % word)
        matching_terms = [''.join(key_chars) for key_chars, value in matches]
        return matching_terms

    @staticmethod
    def create_vocab_trie(embedding):
        """Create trie with vocab terms of the given embedding to enable quick prefix searches.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which trie is to be created.

        Returns
        -------
        :class:`pygtrie.Trie`
            Trie containing vocab terms of the input embedding.

        """
        try:
            from pygtrie import Trie
        except ImportError:
            raise ImportError('pygtrie could not be imported, please install pygtrie in order to use LexicalEntailmentEvaluation')
        vocab_trie = Trie()
        for key in embedding.key_to_index:
            vocab_trie[key] = True
        return vocab_trie

    def evaluate_spearman(self, embedding):
        """Evaluate spearman scores for lexical entailment for given embedding.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which evaluation is to be done.

        Returns
        -------
        float
            Spearman correlation score for the task for input embedding.

        """
        predicted_scores = []
        expected_scores = []
        skipped = 0
        count = 0
        vocab_trie = self.create_vocab_trie(embedding)
        for (word_1, word_2), expected_score in self.scores.items():
            try:
                predicted_score = self.score_function(embedding, vocab_trie, word_1, word_2)
            except ValueError:
                skipped += 1
                continue
            count += 1
            predicted_scores.append(predicted_score)
            expected_scores.append(expected_score)
        logger.info('skipped pairs: %d out of %d' % (skipped, len(self.scores)))
        spearman = spearmanr(expected_scores, predicted_scores)
        return spearman.correlation