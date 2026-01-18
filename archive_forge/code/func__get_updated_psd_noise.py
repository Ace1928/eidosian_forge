import warnings
from typing import Optional, Union
import torch
from torch import Tensor
from torchaudio import functional as F
def _get_updated_psd_noise(self, psd_n: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
    """Update psd of noise recursively.

        Args:
            psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            mask_n (torch.Tensor or None, optional): Time-Frequency mask of the noise.
                Tensor with dimensions `(..., freq, time)`.

        Returns:
            torch.Tensor:  The updated PSD matrix of noise.
        """
    numerator = self.mask_sum_n / (self.mask_sum_n + mask_n.sum(dim=-1))
    denominator = 1 / (self.mask_sum_n + mask_n.sum(dim=-1))
    psd_n = self.psd_n * numerator[..., None, None] + psd_n * denominator[..., None, None]
    return psd_n