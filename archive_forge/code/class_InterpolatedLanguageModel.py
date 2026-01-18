from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey, WittenBell
class InterpolatedLanguageModel(LanguageModel):
    """Logic common to all interpolated language models.

    The idea to abstract this comes from Chen & Goodman 1995.
    Do not instantiate this class directly!
    """

    def __init__(self, smoothing_cls, order, **kwargs):
        params = kwargs.pop('params', {})
        super().__init__(order, **kwargs)
        self.estimator = smoothing_cls(self.vocab, self.counts, **params)

    def unmasked_score(self, word, context=None):
        if not context:
            return self.estimator.unigram_score(word)
        if not self.counts[context]:
            alpha, gamma = (0, 1)
        else:
            alpha, gamma = self.estimator.alpha_gamma(word, context)
        return alpha + gamma * self.unmasked_score(word, context[1:])