import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
def _adjust_coeff(coeffs: Union[float, torch.Tensor], name: str) -> torch.Tensor:
    """Validates and converts absorption or scattering parameters to a tensor with appropriate shape

    Args:
        coeff (float or torch.Tensor): The absorption coefficients of wall materials.

            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.

            If ``absorption`` is a 1D Tensor, the shape must be `(2*dim,)`,
            where the values represent absorption coefficients of ``"west"``, ``"east"``,
            ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.

            If ``absorption`` is a 2D Tensor, the shape must be `(7, 2*dim)`,
            where 7 represents the number of octave bands.

    Returns:
        (torch.Tensor): The expanded coefficient.
            The shape is `(1, 6)` for single octave band case, and
            `(7, 6)` for multi octave band case.
    """
    num_walls = 6
    if isinstance(coeffs, float):
        if coeffs < 0:
            raise ValueError(f'`{name}` must be non-negative. Found: {coeffs}')
        return torch.full((1, num_walls), coeffs)
    if isinstance(coeffs, Tensor):
        if torch.any(coeffs < 0):
            raise ValueError(f'`{name}` must be non-negative. Found: {coeffs}')
        if coeffs.ndim == 1:
            if coeffs.numel() != num_walls:
                raise ValueError(f'The shape of `{name}` must be ({num_walls},) when it is a 1D Tensor. Found the shape {coeffs.shape}.')
            return coeffs.unsqueeze(0)
        if coeffs.ndim == 2:
            if coeffs.shape[1] != num_walls:
                raise ValueError(f'The shape of `{name}` must be (NUM_BANDS, {num_walls}) when it is a 2D Tensor. Found: {coeffs.shape}.')
            return coeffs
    raise TypeError(f'`{name}` must be float or Tensor.')