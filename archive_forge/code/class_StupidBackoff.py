from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey, WittenBell
class StupidBackoff(LanguageModel):
    """Provides StupidBackoff scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a parameter alpha with which we scale the lower order probabilities.
    Note that this is not a true probability distribution as scores for ngrams
    of the same order do not sum up to unity.
    """

    def __init__(self, alpha=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def unmasked_score(self, word, context=None):
        if not context:
            return self.counts.unigrams.freq(word)
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        if word_count > 0:
            return word_count / norm_count
        else:
            return self.alpha * self.unmasked_score(word, context[1:])