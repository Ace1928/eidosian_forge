from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
class ROCAUCScore:
    """An AUC ROC score. This is only defined for binary classification.
    Use the method is_binary before calculating the score, otherwise it
    may throw an error."""

    def __init__(self) -> None:
        self.golds: List[Any] = []
        self.cands: List[Any] = []
        self.saved_score = 0.0
        self.saved_score_at_len = 0

    def score_set(self, cand, gold) -> None:
        self.cands.append(cand)
        self.golds.append(gold)

    def is_binary(self):
        return len(np.unique(self.golds)) == 2

    @property
    def score(self):
        if not self.is_binary():
            raise ValueError(Errors.E165.format(label=set(self.golds)))
        if len(self.golds) == self.saved_score_at_len:
            return self.saved_score
        self.saved_score = _roc_auc_score(self.golds, self.cands)
        self.saved_score_at_len = len(self.golds)
        return self.saved_score