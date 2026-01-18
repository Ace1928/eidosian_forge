import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
def _packed_sequence_init_args(data: Tensor, batch_sizes: Optional[Tensor]=None, sorted_indices: Optional[Tensor]=None, unsorted_indices: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    if unsorted_indices is None:
        unsorted_indices = invert_permutation(sorted_indices)
    if batch_sizes is not None:
        if batch_sizes.device.type != 'cpu':
            raise ValueError('batch_sizes should always be on CPU. Instances of PackedSequence should never be created manually. They should be instantiated by functions like pack_sequence and pack_padded_sequences in nn.utils.rnn. https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence')
        return (data, batch_sizes, sorted_indices, unsorted_indices)
    else:
        assert isinstance(data, (list, tuple)) and len(data) == 2
        return (data[0], data[1], sorted_indices, unsorted_indices)