import typing
from typing import Any, Callable, List, Union, cast, Sequence
import torch
from torch import Tensor
import torch.cuda.comm
def _setitem_by_slice(self, index: slice, value) -> None:
    if not index.start is index.stop is index.step is None:
        raise NotImplementedError('only slice [:] supported')
    if not self.atomic:
        self._values = value
        return
    if len(value) != 1:
        raise IndexError('atomic batch cannot be replaced with multiple tensors')
    self._values = value[0]