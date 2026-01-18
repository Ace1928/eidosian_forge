import collections
import warnings
from typing import Optional, Sequence, Union
import torch.cuda
def _check_sequence_type(inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
    if not isinstance(inputs, collections.abc.Container) or isinstance(inputs, torch.Tensor):
        raise TypeError('Inputs should be a collection of tensors')