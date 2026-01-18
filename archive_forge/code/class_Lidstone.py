from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey, WittenBell
class Lidstone(LanguageModel):
    """Provides Lidstone-smoothed scores.

    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """

    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def unmasked_score(self, word, context=None):
        """Add-one smoothing: Lidstone or Laplace.

        To see what kind, look at `gamma` attribute on the class.

        """
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        return (word_count + self.gamma) / (norm_count + len(self.vocab) * self.gamma)