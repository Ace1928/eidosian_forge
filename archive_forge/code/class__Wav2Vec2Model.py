from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
class _Wav2Vec2Model(nn.Module):
    """Wrapper class for :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This is used for layer normalization at the input
    """

    def __init__(self, model: Wav2Vec2Model, normalize_waveform: bool, apply_log_softmax: bool, append_star: bool):
        super().__init__()
        self.model = model
        self.normalize_waveform = normalize_waveform
        self.apply_log_softmax = apply_log_softmax
        self.append_star = append_star

    def forward(self, waveforms: Tensor, lengths: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.normalize_waveform:
            waveforms = nn.functional.layer_norm(waveforms, waveforms.shape)
        output, output_lengths = self.model(waveforms, lengths)
        if self.apply_log_softmax:
            output = torch.nn.functional.log_softmax(output, dim=-1)
        if self.append_star:
            star_dim = torch.zeros((1, output.size(1), 1), dtype=output.dtype, device=output.device)
            output = torch.cat((output, star_dim), dim=-1)
        return (output, output_lengths)

    @torch.jit.export
    def extract_features(self, waveforms: Tensor, lengths: Optional[Tensor]=None, num_layers: Optional[int]=None) -> Tuple[List[Tensor], Optional[Tensor]]:
        if self.normalize_waveform:
            waveforms = nn.functional.layer_norm(waveforms, waveforms.shape)
        return self.model.extract_features(waveforms, lengths, num_layers)