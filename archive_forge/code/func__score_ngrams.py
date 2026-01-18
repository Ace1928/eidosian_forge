import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def _score_ngrams(self, score_fn):
    """Generates of (ngram, score) pairs as determined by the scoring
        function provided.
        """
    for tup in self.ngram_fd:
        score = self.score_ngram(score_fn, *tup)
        if score is not None:
            yield (tup, score)