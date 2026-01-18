import warnings
from typing import Optional, Union
import torch
from torch import Tensor
from torchaudio import functional as F

        Args:
            specgram (torch.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`.
            psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            reference_channel (int or torch.Tensor): Specifies the reference channel.
                If the dtype is ``int``, it represents the reference channel index.
                If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
                is one-hot.
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            torch.Tensor: Single-channel complex-valued enhanced spectrum with dimensions `(..., freq, time)`.
        