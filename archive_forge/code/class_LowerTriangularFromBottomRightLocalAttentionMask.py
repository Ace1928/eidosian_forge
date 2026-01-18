import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class LowerTriangularFromBottomRightLocalAttentionMask(LowerTriangularFromBottomRightMask):
    """
    A mask that combines both :attr:`LowerTriangularFromBottomRightMask` and
    local attention.

    A query whose distance from the final query is X cannot attend to a key
    whose distance to the final key is either of:

    * less than X (i.e. "causal attention", same as :attr:`LowerTriangularFromBottomRightMask`)
    * greater than X + window_size (i.e. "local attention")


    .. figure:: /_static/causal_bottom_right_local.png

        The mask from :attr:`LowerTriangularFromBottomRightLocalAttentionMask`.
        The green area is calculated, and the grey area is masked out.
    """
    _window_size: int

    def __post_init__(self) -> None:
        if self._window_size <= 0:
            raise ValueError(f'Expected `window_size > 0`, but window_size={self._window_size}')

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        return _materialize_causal_mask(shape, dtype=dtype, device=device, window_size=self._window_size, from_bottomright=True)