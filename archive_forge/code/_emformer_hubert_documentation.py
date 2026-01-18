from typing import List, Optional, Tuple
import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction
Extract output Tensors of the emformer layers.

        Args:
            input (torch.Tensor): The input feature for emformer encoder.
                Tensor with dimensions `(batch, time, feature_dim)`.
            lengths (torch.Tensor or None): Valid length of each input sample.
                Tensor with dimension `(batch, )`.
            num_layers (int or None, optional): If not ``None``, returns the first
                `num_layers` layers of Tensors as the output, otherwise returns the
                Tensors from all emformer layers.

        Returns:
            List[torch.Tensor]:
                Output Tensors of selected emformer layers.
        