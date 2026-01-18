from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey, WittenBell
class KneserNeyInterpolated(InterpolatedLanguageModel):
    """Interpolated version of Kneser-Ney smoothing."""

    def __init__(self, order, discount=0.1, **kwargs):
        if not 0 <= discount <= 1:
            raise ValueError('Discount must be between 0 and 1 for probabilities to sum to unity.')
        super().__init__(KneserNey, order, params={'discount': discount, 'order': order}, **kwargs)