import math
from dataclasses import dataclass
from typing import (
import torch
class LowerTriangularMask(AttentionBias):
    """
    A lower-triangular (aka causal) mask

    A query Q cannot attend to a key which is farther from the
    initial key than Q is from the initial query.

    See also :attr:`LowerTriangularFromBottomRightMask` if the number
    of queries is not equal to the number of keys/values.
    """

    def __init__(self, *tensor_args, **tensor_kwargs) -> None:
        super().__init__()

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return _materialize_causal_mask(shape, dtype=dtype, device=device)

    def add_bias(self, bias: torch.Tensor) -> 'LowerTriangularMaskWithTensorBias':
        """
        Creates a new causal mask with an arbitrary ``torch.Tensor`` bias
        """
        return LowerTriangularMaskWithTensorBias(bias)