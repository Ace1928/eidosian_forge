import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _get_unigrams(sentences):
    """
        Calcualtes uni-grams.

        Args:
          sentences: list of string

        Returns:
          A set of n-grams and their freqneucy
        """
    assert len(sentences) > 0
    tokens = Rouge._split_into_words(sentences)
    unigram_set = collections.defaultdict(int)
    for token in tokens:
        unigram_set[token] += 1
    return (unigram_set, len(tokens))