import warnings
from typing import Optional, Union
import torch
from torch import Tensor
from torchaudio import functional as F
def _get_updated_psd_speech(self, psd_s: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
    """Update psd of speech recursively.

        Args:
            psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            mask_s (torch.Tensor): Time-Frequency mask of the target speech.
                Tensor with dimensions `(..., freq, time)`.

        Returns:
            torch.Tensor: The updated PSD matrix of target speech.
        """
    numerator = self.mask_sum_s / (self.mask_sum_s + mask_s.sum(dim=-1))
    denominator = 1 / (self.mask_sum_s + mask_s.sum(dim=-1))
    psd_s = self.psd_s * numerator[..., None, None] + psd_s * denominator[..., None, None]
    return psd_s