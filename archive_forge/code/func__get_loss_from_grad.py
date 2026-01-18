from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
def _get_loss_from_grad(self, grads: Sequence[Floats2d]) -> float:
    loss = 0.0
    for grad in grads:
        loss += self.cc._get_loss_from_grad(grad)
    return loss