from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import AbsoluteDiscounting, KneserNey, WittenBell
class AbsoluteDiscountingInterpolated(InterpolatedLanguageModel):
    """Interpolated version of smoothing with absolute discount."""

    def __init__(self, order, discount=0.75, **kwargs):
        super().__init__(AbsoluteDiscounting, order, params={'discount': discount}, **kwargs)