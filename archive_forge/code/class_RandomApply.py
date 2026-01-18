from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch import nn
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform
class RandomApply(Transform):
    """[BETA] Apply randomly a list of transformations with a given probability.

    .. v2betastatus:: RandomApply transform

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability of applying the list of transforms
    """
    _v1_transform_cls = _transforms.RandomApply

    def __init__(self, transforms: Union[Sequence[Callable], nn.ModuleList], p: float=0.5) -> None:
        super().__init__()
        if not isinstance(transforms, (Sequence, nn.ModuleList)):
            raise TypeError('Argument transforms should be a sequence of callables or a `nn.ModuleList`')
        self.transforms = transforms
        if not 0.0 <= p <= 1.0:
            raise ValueError('`p` should be a floating point value in the interval [0.0, 1.0].')
        self.p = p

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        return {'transforms': self.transforms, 'p': self.p}

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return sample
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f'    {t}')
        return '\n'.join(format_string)