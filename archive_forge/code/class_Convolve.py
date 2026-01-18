import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
class Convolve(torch.nn.Module):
    """
    Convolves inputs along their last dimension using the direct method.
    Note that, in contrast to :class:`torch.nn.Conv1d`, which actually applies the valid cross-correlation
    operator, this module applies the true `convolution`_ operator.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`, where
              `N` and `M` are the trailing dimensions of the two inputs. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """

    def __init__(self, mode: str='full') -> None:
        _check_convolve_mode(mode)
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): First convolution operand, with shape `(..., N)`.
            y (torch.Tensor): Second convolution operand, with shape `(..., M)`
                (leading dimensions must be broadcast-able with those of ``x``).

        Returns:
            torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
            the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.
        """
        return F.convolve(x, y, mode=self.mode)