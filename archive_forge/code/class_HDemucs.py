import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class HDemucs(torch.nn.Module):
    """Hybrid Demucs model from
    *Hybrid Spectrogram and Waveform Source Separation* :cite:`defossez2021hybrid`.

    See Also:
        * :class:`torchaudio.pipelines.SourceSeparationBundle`: Source separation pipeline with pre-trained models.

    Args:
        sources (List[str]): list of source names. List can contain the following source
            options: [``"bass"``, ``"drums"``, ``"other"``, ``"mixture"``, ``"vocals"``].
        audio_channels (int, optional): input/output audio channels. (Default: 2)
        channels (int, optional): initial number of hidden channels. (Default: 48)
        growth (int, optional): increase the number of hidden channels by this factor at each layer. (Default: 2)
        nfft (int, optional): number of fft bins. Note that changing this requires careful computation of
            various shape parameters and will not work out of the box for hybrid models. (Default: 4096)
        depth (int, optional): number of layers in encoder and decoder (Default: 6)
        freq_emb (float, optional): add frequency embedding after the first frequency layer if > 0,
            the actual value controls the weight of the embedding. (Default: 0.2)
        emb_scale (int, optional): equivalent to scaling the embedding learning rate (Default: 10)
        emb_smooth (bool, optional): initialize the embedding with a smooth one (with respect to frequencies).
            (Default: ``True``)
        kernel_size (int, optional): kernel_size for encoder and decoder layers. (Default: 8)
        time_stride (int, optional): stride for the final time layer, after the merge. (Default: 2)
        stride (int, optional): stride for encoder and decoder layers. (Default: 4)
        context (int, optional): context for 1x1 conv in the decoder. (Default: 4)
        context_enc (int, optional): context for 1x1 conv in the encoder. (Default: 0)
        norm_starts (int, optional): layer at which group norm starts being used.
            decoder layers are numbered in reverse order. (Default: 4)
        norm_groups (int, optional): number of groups for group norm. (Default: 4)
        dconv_depth (int, optional): depth of residual DConv branch. (Default: 2)
        dconv_comp (int, optional): compression of DConv branch. (Default: 4)
        dconv_attn (int, optional): adds attention layers in DConv branch starting at this layer. (Default: 4)
        dconv_lstm (int, optional): adds a LSTM layer in DConv branch starting at this layer. (Default: 4)
        dconv_init (float, optional): initial scale for the DConv branch LayerScale. (Default: 1e-4)
    """

    def __init__(self, sources: List[str], audio_channels: int=2, channels: int=48, growth: int=2, nfft: int=4096, depth: int=6, freq_emb: float=0.2, emb_scale: int=10, emb_smooth: bool=True, kernel_size: int=8, time_stride: int=2, stride: int=4, context: int=1, context_enc: int=0, norm_starts: int=4, norm_groups: int=4, dconv_depth: int=2, dconv_comp: int=4, dconv_attn: int=4, dconv_lstm: int=4, dconv_init: float=0.0001):
        super().__init__()
        self.depth = depth
        self.nfft = nfft
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.channels = channels
        self.hop_length = self.nfft // 4
        self.freq_emb = None
        self.freq_encoder = nn.ModuleList()
        self.freq_decoder = nn.ModuleList()
        self.time_encoder = nn.ModuleList()
        self.time_decoder = nn.ModuleList()
        chin = audio_channels
        chin_z = chin * 2
        chout = channels
        chout_z = channels
        freqs = self.nfft // 2
        for index in range(self.depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
            norm_type = 'group_norm' if index >= norm_starts else 'none'
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                if freqs != 1:
                    raise ValueError('When freq is false, freqs must be 1.')
                ker = time_stride * 2
                stri = time_stride
            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True
            kw = {'kernel_size': ker, 'stride': stri, 'freq': freq, 'pad': pad, 'norm_type': norm_type, 'norm_groups': norm_groups, 'dconv_kw': {'lstm': lstm, 'attn': attn, 'depth': dconv_depth, 'compress': dconv_comp, 'init': dconv_init}}
            kwt = dict(kw)
            kwt['freq'] = 0
            kwt['kernel_size'] = kernel_size
            kwt['stride'] = stride
            kwt['pad'] = True
            kw_dec = dict(kw)
            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z
            enc = _HEncLayer(chin_z, chout_z, context=context_enc, **kw)
            if freq:
                if last_freq is True and nfft == 2048:
                    kwt['stride'] = 2
                    kwt['kernel_size'] = 4
                tenc = _HEncLayer(chin, chout, context=context_enc, empty=last_freq, **kwt)
                self.time_encoder.append(tenc)
            self.freq_encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * 2
            dec = _HDecLayer(chout_z, chin_z, last=index == 0, context=context, **kw_dec)
            if freq:
                tdec = _HDecLayer(chout, chin, empty=last_freq, last=index == 0, context=context, **kwt)
                self.time_decoder.insert(0, tdec)
            self.freq_decoder.insert(0, dec)
            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = _ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb
        _rescale_module(self)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x
        if hl != nfft // 4:
            raise ValueError('Hop length must be nfft // 4')
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = self._pad1d(x, pad, pad + le * hl - x.shape[-1], mode='reflect')
        z = _spectro(x, nfft, hl)[..., :-1, :]
        if z.shape[-1] != le + 4:
            raise ValueError("Spectrogram's last dimension must be 4 + input size divided by stride")
        z = z[..., 2:2 + le]
        return z

    def _ispec(self, z, length=None):
        hl = self.hop_length
        z = F.pad(z, [0, 0, 0, 1])
        z = F.pad(z, [2, 2])
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = _ispectro(z, hl, length=le)
        x = x[..., pad:pad + length]
        return x

    def _pad1d(self, x: torch.Tensor, padding_left: int, padding_right: int, mode: str='zero', value: float=0.0):
        """Wrapper around F.pad, in order for reflect padding when num_frames is shorter than max_pad.
        Add extra zero padding around in order for padding to not break."""
        length = x.shape[-1]
        if mode == 'reflect':
            max_pad = max(padding_left, padding_right)
            if length <= max_pad:
                x = F.pad(x, (0, max_pad - length + 1))
        return F.pad(x, (padding_left, padding_right), mode, value)

    def _magnitude(self, z):
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _mask(self, m):
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out

    def forward(self, input: torch.Tensor):
        """HDemucs forward call

        Args:
            input (torch.Tensor): input mixed tensor of shape `(batch_size, channel, num_frames)`

        Returns:
            Tensor
                output tensor split into sources of shape `(batch_size, num_sources, channel, num_frames)`
        """
        if input.ndim != 3:
            raise ValueError(f'Expected 3D tensor with dimensions (batch, channel, frames). Found: {input.shape}')
        if input.shape[1] != self.audio_channels:
            raise ValueError(f'The channel dimension of input Tensor must match `audio_channels` of HDemucs model. Found:{input.shape[1]}.')
        x = input
        length = x.shape[-1]
        z = self._spec(input)
        mag = self._magnitude(z)
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-05 + std)
        xt = input
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-05 + stdt)
        saved = []
        saved_t = []
        lengths: List[int] = []
        lengths_t: List[int] = []
        for idx, encode in enumerate(self.freq_encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.time_encoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.time_encoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)
        x = torch.zeros_like(x)
        xt = torch.zeros_like(x)
        for idx, decode in enumerate(self.freq_decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.time_decoder)
            if idx >= offset:
                tdec = self.time_decoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    if pre.shape[2] != 1:
                        raise ValueError(f'If tdec empty is True, pre shape does not match {pre.shape}')
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        if len(saved) != 0:
            raise AssertionError('saved is not empty')
        if len(lengths_t) != 0:
            raise AssertionError('lengths_t is not empty')
        if len(saved_t) != 0:
            raise AssertionError('saved_t is not empty')
        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        zout = self._mask(x)
        x = self._ispec(zout, length)
        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        return x