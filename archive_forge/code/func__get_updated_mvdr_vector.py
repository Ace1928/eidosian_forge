import warnings
from typing import Optional, Union
import torch
from torch import Tensor
from torchaudio import functional as F
def _get_updated_mvdr_vector(self, psd_s: torch.Tensor, psd_n: torch.Tensor, mask_s: torch.Tensor, mask_n: torch.Tensor, reference_vector: torch.Tensor, solution: str='ref_channel', diagonal_loading: bool=True, diag_eps: float=1e-07, eps: float=1e-08) -> torch.Tensor:
    """Recursively update the MVDR beamforming vector.

        Args:
            psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            mask_s (torch.Tensor): Time-Frequency mask of the target speech.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False``
                or with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
            mask_n (torch.Tensor or None, optional): Time-Frequency mask of the noise.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False``
                or with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
            reference_vector (torch.Tensor): One-hot reference channel matrix.
            solution (str, optional): Solution to compute the MVDR beamforming weights.
                Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            torch.Tensor: The MVDR beamforming weight matrix.
        """
    if self.multi_mask:
        mask_s = mask_s.mean(dim=-3)
        mask_n = mask_n.mean(dim=-3)
    if self.psd_s.ndim == 1:
        self.psd_s = psd_s
        self.psd_n = psd_n
        self.mask_sum_s = mask_s.sum(dim=-1)
        self.mask_sum_n = mask_n.sum(dim=-1)
        return _get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)
    else:
        psd_s = self._get_updated_psd_speech(psd_s, mask_s)
        psd_n = self._get_updated_psd_noise(psd_n, mask_n)
        self.psd_s = psd_s
        self.psd_n = psd_n
        self.mask_sum_s = self.mask_sum_s + mask_s.sum(dim=-1)
        self.mask_sum_n = self.mask_sum_n + mask_n.sum(dim=-1)
        return _get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)