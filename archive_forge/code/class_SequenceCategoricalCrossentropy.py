from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
class SequenceCategoricalCrossentropy(Loss):

    def __init__(self, *, normalize: bool=True, names: Optional[Sequence[str]]=None, missing_value: Optional[Union[str, int]]=None, neg_prefix: Optional[str]=None, label_smoothing: float=0.0):
        self.cc = CategoricalCrossentropy(normalize=False, names=names, missing_value=missing_value, neg_prefix=neg_prefix, label_smoothing=label_smoothing)
        self.normalize = normalize

    def __call__(self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]) -> Tuple[List[Floats2d], float]:
        grads = self.get_grad(guesses, truths)
        loss = self._get_loss_from_grad(grads)
        return (grads, loss)

    def get_grad(self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]) -> List[Floats2d]:
        err = 'Cannot calculate SequenceCategoricalCrossentropy loss: guesses and truths must be same length'
        if len(guesses) != len(truths):
            raise ValueError(err)
        n = len(guesses)
        d_scores = []
        for yh, y in zip(guesses, truths):
            d_yh = self.cc.get_grad(yh, y)
            if self.normalize:
                d_yh /= n
            d_scores.append(d_yh)
        return d_scores

    def get_loss(self, guesses: Sequence[Floats2d], truths: Sequence[IntsOrFloatsOrStrs]) -> float:
        return self._get_loss_from_grad(self.get_grad(guesses, truths))

    def _get_loss_from_grad(self, grads: Sequence[Floats2d]) -> float:
        loss = 0.0
        for grad in grads:
            loss += self.cc._get_loss_from_grad(grad)
        return loss