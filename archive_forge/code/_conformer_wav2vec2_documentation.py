from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components

        Args:
            features (Tensor):
                Tensor of audio features of shape `(batch, frame, dim)`.
            audio_lengths (Tensor or None, optional):
                Tensor of valid length of each valid auidio in the batch.
                shape: `(batch, )` (Default: ``None``)

        Returns:
            (Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor):
            Tensor
                The masked sequences of probability distribution of shape `(batch, frame dim)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )` representing
                valid length in time axis is returns.
            Tensor
                The mask indices.
            Tensor
                The targets, prior to negative sampling.
            Tensor
                The negative samples.
            Tensor
                The indices of the negative samples.
        