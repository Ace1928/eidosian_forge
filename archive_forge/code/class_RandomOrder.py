from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch import nn
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform
class RandomOrder(Transform):
    """[BETA] Apply a list of transformations in a random order.

    .. v2betastatus:: RandomOrder transform

    This transform does not support torchscript.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError('Argument transforms should be a sequence of callables')
        super().__init__()
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for idx in torch.randperm(len(self.transforms)):
            transform = self.transforms[idx]
            sample = transform(sample)
        return sample