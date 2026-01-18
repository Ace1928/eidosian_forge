import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class LocalAttentionFromBottomRightMask(AttentionBias):
    """
    A local attention mask

    The query at position :math:`q` can attend the key at position :math:`k` if
    :math:`q - window\\_left <= k + s <= q + window\\_right`

    With :math:`s = num\\_queries - num\\_keys`

    :Example:

    .. code-block:: python

        import torch
        from xformers.ops import fmha

        bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=1, window_right=2)
        print(bias.materialize(shape=(4, 4)).exp())
        print(bias.materialize(shape=(4, 5)).exp())

    .. code-block:: text

        # 4x4
        tensor([[1., 1., 1., 0.],
                [1., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 0., 1., 1.]])

        # 4x5
        tensor([[1., 1., 1., 1., 0.],
                [0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1.]])

    :Illustration:

    .. figure:: /_static/local_attn.png
        :width: 240px

        The total window size is :math:`window\\_left + 1 + window\\_right`
    """
    window_left: int
    window_right: int

    def __post_init__(self) -> None:
        if self.window_left < 0:
            raise ValueError(f'Invalid window value passed to `LocalAttentionFromBottomRightMask`: expected`window_left > 0` but got window_left={self.window_left}')
        if self.window_right < 0:
            raise ValueError(f'Invalid window value passed to `LocalAttentionFromBottomRightMask`: expected`window_right > 0` but got window_right={self.window_right}')

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        create_as = dtype if dtype is not torch.bfloat16 else torch.float32
        mask = torch.full(shape, dtype=create_as, fill_value=1, device=device)
        num_queries, num_keys = shape[-2:]
        shift = num_keys - num_queries
        mask = torch.triu(mask, diagonal=shift - self.window_left)
        mask = torch.tril(mask, diagonal=shift + self.window_right)
        mask = torch.log(mask)
        return mask.to(dtype)