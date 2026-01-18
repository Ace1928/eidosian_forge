import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
class AbstractCollocationFinder:
    """
    An abstract base class for collocation finders whose purpose is to
    collect collocation candidate frequencies, filter and rank them.

    As a minimum, collocation finders require the frequencies of each
    word in a corpus, and the joint frequency of word tuples. This data
    should be provided through nltk.probability.FreqDist objects or an
    identical interface.
    """

    def __init__(self, word_fd, ngram_fd):
        self.word_fd = word_fd
        self.N = word_fd.N()
        self.ngram_fd = ngram_fd

    @classmethod
    def _build_new_documents(cls, documents, window_size, pad_left=False, pad_right=False, pad_symbol=None):
        """
        Pad the document with the place holder according to the window_size
        """
        padding = (pad_symbol,) * (window_size - 1)
        if pad_right:
            return _itertools.chain.from_iterable((_itertools.chain(doc, padding) for doc in documents))
        if pad_left:
            return _itertools.chain.from_iterable((_itertools.chain(padding, doc) for doc in documents))

    @classmethod
    def from_documents(cls, documents):
        """Constructs a collocation finder given a collection of documents,
        each of which is a list (or iterable) of tokens.
        """
        return cls.from_words(cls._build_new_documents(documents, cls.default_ws, pad_right=True))

    @staticmethod
    def _ngram_freqdist(words, n):
        return FreqDist((tuple(words[i:i + n]) for i in range(len(words) - 1)))

    def _apply_filter(self, fn=lambda ngram, freq: False):
        """Generic filter removes ngrams from the frequency distribution
        if the function returns True when passed an ngram tuple.
        """
        tmp_ngram = FreqDist()
        for ngram, freq in self.ngram_fd.items():
            if not fn(ngram, freq):
                tmp_ngram[ngram] = freq
        self.ngram_fd = tmp_ngram

    def apply_freq_filter(self, min_freq):
        """Removes candidate ngrams which have frequency less than min_freq."""
        self._apply_filter(lambda ng, freq: freq < min_freq)

    def apply_ngram_filter(self, fn):
        """Removes candidate ngrams (w1, w2, ...) where fn(w1, w2, ...)
        evaluates to True.
        """
        self._apply_filter(lambda ng, f: fn(*ng))

    def apply_word_filter(self, fn):
        """Removes candidate ngrams (w1, w2, ...) where any of (fn(w1), fn(w2),
        ...) evaluates to True.
        """
        self._apply_filter(lambda ng, f: any((fn(w) for w in ng)))

    def _score_ngrams(self, score_fn):
        """Generates of (ngram, score) pairs as determined by the scoring
        function provided.
        """
        for tup in self.ngram_fd:
            score = self.score_ngram(score_fn, *tup)
            if score is not None:
                yield (tup, score)

    def score_ngrams(self, score_fn):
        """Returns a sequence of (ngram, score) pairs ordered from highest to
        lowest score, as determined by the scoring function provided.
        """
        return sorted(self._score_ngrams(score_fn), key=lambda t: (-t[1], t[0]))

    def nbest(self, score_fn, n):
        """Returns the top n ngrams when scored by the given function."""
        return [p for p, s in self.score_ngrams(score_fn)[:n]]

    def above_score(self, score_fn, min_score):
        """Returns a sequence of ngrams, ordered by decreasing score, whose
        scores each exceed the given minimum score.
        """
        for ngram, score in self.score_ngrams(score_fn):
            if score > min_score:
                yield ngram
            else:
                break