import warnings
from typing import Optional, Union
import torch
from torch import Tensor
from torchaudio import functional as F
class RTFMVDR(torch.nn.Module):
    """Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) module
    based on the relative transfer function (RTF) and power spectral density (PSD) matrix of noise.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the multi-channel complex-valued spectrum :math:`\\textbf{Y}`, the relative transfer function (RTF) matrix
    or the steering vector of target speech :math:`\\bm{v}`, the PSD matrix of noise :math:`\\bf{\\Phi}_{\\textbf{NN}}`, and
    a one-hot vector that represents the reference channel :math:`\\bf{u}`, the module computes the single-channel
    complex-valued spectrum of the enhanced speech :math:`\\hat{\\textbf{S}}`. The formula is defined as:

    .. math::
        \\hat{\\textbf{S}}(f) = \\textbf{w}_{\\text{bf}}(f)^{\\mathsf{H}} \\textbf{Y}(f)

    where :math:`\\textbf{w}_{\\text{bf}}(f)` is the MVDR beamforming weight for the :math:`f`-th frequency bin,
    :math:`(.)^{\\mathsf{H}}` denotes the Hermitian Conjugate operation.

    The beamforming weight is computed by:

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}}
        {{\\bm{v}^{\\mathsf{H}}}(f){\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}
    """

    def forward(self, specgram: Tensor, rtf: Tensor, psd_n: Tensor, reference_channel: Union[int, Tensor], diagonal_loading: bool=True, diag_eps: float=1e-07, eps: float=1e-08) -> Tensor:
        """
        Args:
            specgram (torch.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`
            rtf (torch.Tensor): The complex-valued RTF vector of target speech.
                Tensor with dimensions `(..., freq, channel)`.
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
        """
        w_mvdr = F.mvdr_weights_rtf(rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps)
        spectrum_enhanced = F.apply_beamforming(w_mvdr, specgram)
        return spectrum_enhanced