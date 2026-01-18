from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
class L2Distance(Loss):

    def __init__(self, *, normalize: bool=True):
        self.normalize = normalize

    def __call__(self, guesses: Floats2d, truths: Floats2d) -> Tuple[Floats2d, float]:
        return (self.get_grad(guesses, truths), self.get_loss(guesses, truths))

    def get_grad(self, guesses: Floats2d, truths: Floats2d) -> Floats2d:
        if guesses.shape != truths.shape:
            err = f'Cannot calculate L2 distance: mismatched shapes: {guesses.shape} vs {truths.shape}.'
            raise ValueError(err)
        difference = guesses - truths
        if self.normalize:
            difference = difference / guesses.shape[0]
        return difference

    def get_loss(self, guesses: Floats2d, truths: Floats2d) -> float:
        if guesses.shape != truths.shape:
            err = f'Cannot calculate L2 distance: mismatched shapes: {guesses.shape} vs {truths.shape}.'
            raise ValueError(err)
        d_truth = self.get_grad(guesses, truths)
        return (d_truth ** 2).sum()