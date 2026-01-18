import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
class ContextIndex:
    """
    A bidirectional index between words and their 'contexts' in a text.
    The context of a word is usually defined to be the words that occur
    in a fixed window around the word; but other definitions may also
    be used by providing a custom context function.
    """

    @staticmethod
    def _default_context(tokens, i):
        """One left token and one right token, normalized to lowercase"""
        left = tokens[i - 1].lower() if i != 0 else '*START*'
        right = tokens[i + 1].lower() if i != len(tokens) - 1 else '*END*'
        return (left, right)

    def __init__(self, tokens, context_func=None, filter=None, key=lambda x: x):
        self._key = key
        self._tokens = tokens
        if context_func:
            self._context_func = context_func
        else:
            self._context_func = self._default_context
        if filter:
            tokens = [t for t in tokens if filter(t)]
        self._word_to_contexts = CFD(((self._key(w), self._context_func(tokens, i)) for i, w in enumerate(tokens)))
        self._context_to_words = CFD(((self._context_func(tokens, i), self._key(w)) for i, w in enumerate(tokens)))

    def tokens(self):
        """
        :rtype: list(str)
        :return: The document that this context index was
            created from.
        """
        return self._tokens

    def word_similarity_dict(self, word):
        """
        Return a dictionary mapping from words to 'similarity scores,'
        indicating how often these two words occur in the same
        context.
        """
        word = self._key(word)
        word_contexts = set(self._word_to_contexts[word])
        scores = {}
        for w, w_contexts in self._word_to_contexts.items():
            scores[w] = f_measure(word_contexts, set(w_contexts))
        return scores

    def similar_words(self, word, n=20):
        scores = defaultdict(int)
        for c in self._word_to_contexts[self._key(word)]:
            for w in self._context_to_words[c]:
                if w != word:
                    scores[w] += self._context_to_words[c][word] * self._context_to_words[c][w]
        return sorted(scores, key=scores.get, reverse=True)[:n]

    def common_contexts(self, words, fail_on_unknown=False):
        """
        Find contexts where the specified words can all appear; and
        return a frequency distribution mapping each context to the
        number of times that context was used.

        :param words: The words used to seed the similarity search
        :type words: str
        :param fail_on_unknown: If true, then raise a value error if
            any of the given words do not occur at all in the index.
        """
        words = [self._key(w) for w in words]
        contexts = [set(self._word_to_contexts[w]) for w in words]
        empty = [words[i] for i in range(len(words)) if not contexts[i]]
        common = reduce(set.intersection, contexts)
        if empty and fail_on_unknown:
            raise ValueError('The following word(s) were not found:', ' '.join(words))
        elif not common:
            return FreqDist()
        else:
            fd = FreqDist((c for w in words for c in self._word_to_contexts[w] if c in common))
            return fd