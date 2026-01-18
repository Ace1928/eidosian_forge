from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
class SquimSubjective(nn.Module):
    """Speech Quality and Intelligibility Measures (SQUIM) model that predicts **subjective** metric scores
    for speech enhancement (e.g., Mean Opinion Score (MOS)). The model is adopted from *NORESQA-MOS*
    :cite:`manocha2022speech` which predicts MOS scores given the input speech and a non-matching reference.

    Args:
        ssl_model (torch.nn.Module): The self-supervised learning model for feature extraction.
        projector (torch.nn.Module): Projection layer that projects SSL feature to a lower dimension.
        predictor (torch.nn.Module): Predict the subjective scores.
    """

    def __init__(self, ssl_model: nn.Module, projector: nn.Module, predictor: nn.Module):
        super(SquimSubjective, self).__init__()
        self.ssl_model = ssl_model
        self.projector = projector
        self.predictor = predictor

    def _align_shapes(self, waveform: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cut or pad the reference Tensor to make it aligned with waveform Tensor.

        Args:
            waveform (torch.Tensor): Input waveform for evaluation. Tensor with dimensions `(batch, time)`.
            reference (torch.Tensor): Non-matching clean reference. Tensor with dimensions `(batch, time_ref)`.

        Returns:
            (torch.Tensor, torch.Tensor): The aligned waveform and reference Tensors
                with same dimensions `(batch, time)`.
        """
        T_waveform = waveform.shape[-1]
        T_reference = reference.shape[-1]
        if T_reference < T_waveform:
            num_padding = T_waveform // T_reference + 1
            reference = torch.cat([reference for _ in range(num_padding)], dim=1)
        return (waveform, reference[:, :T_waveform])

    def forward(self, waveform: torch.Tensor, reference: torch.Tensor):
        """Predict subjective evaluation metric score.

        Args:
            waveform (torch.Tensor): Input waveform for evaluation. Tensor with dimensions `(batch, time)`.
            reference (torch.Tensor): Non-matching clean reference. Tensor with dimensions `(batch, time_ref)`.

        Returns:
            (torch.Tensor): Subjective metric score. Tensor with dimensions `(batch,)`.
        """
        waveform, reference = self._align_shapes(waveform, reference)
        waveform = self.projector(self.ssl_model.extract_features(waveform)[0][-1])
        reference = self.projector(self.ssl_model.extract_features(reference)[0][-1])
        concat = torch.cat((reference, waveform), dim=2)
        score_diff = self.predictor(concat)
        return 5 - score_diff