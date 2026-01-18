import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def _tik_reg(mat: torch.Tensor, reg: float=1e-07, eps: float=1e-08) -> torch.Tensor:
    """Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (torch.Tensor): Input matrix with dimensions `(..., channel, channel)`.
        reg (float, optional): Regularization factor. (Default: 1e-8)
        eps (float, optional): Value to avoid the correlation matrix is all-zero. (Default: ``1e-8``)

    Returns:
        Tensor: Regularized matrix with dimensions `(..., channel, channel)`.
    """
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    epsilon = _compute_mat_trace(mat).real[..., None, None] * reg
    epsilon = epsilon + eps
    mat = mat + epsilon * eye[..., :, :]
    return mat