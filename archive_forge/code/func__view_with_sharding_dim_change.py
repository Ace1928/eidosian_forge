from typing import Tuple, Union, Sequence, cast
import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor import DTensor as DT
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import (
def _view_with_sharding_dim_change(tensor: Union[torch.Tensor, DT], sharding_dim: int, shape: Tuple[int, ...]) -> Union[torch.Tensor, DT]:
    """
    We change the implicit sharding dim for a distributed tensor without comms.
    Because if we don't change sharding dim, we will ended up having more comms that are not necessary.
    Note that this op will produce invalid DTensor, you will need to call this op in pair to recover
    it back to a valid DTensor.

    This should only be used when implicitly changing sharding dim doesn't have semantic issue.
    """
    if isinstance(tensor, DT):
        return _ViewAndRedistribute.apply(tensor, sharding_dim, shape)
    else:
        return tensor.view(shape)