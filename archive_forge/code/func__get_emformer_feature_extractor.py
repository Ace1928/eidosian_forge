from typing import List, Optional, Tuple
import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction
def _get_emformer_feature_extractor(input_dim: int, output_dim: int, use_bias: bool, stride: int) -> FeatureEncoder:
    """Construct FeatureEncoder for emformer model.

    Args:
        input_dim (int): The feature dimension of log-mel spectrogram feature.
        output_dim (int): The feature dimension after linear layer.
        use_bias (bool): If ``True``, enable bias parameter in the linear layer.
        stride (int): Number of frames to merge for the output frame.

    Returns:
        FeatureEncoder: The resulting FeatureEncoder module.
    """
    return FeatureEncoder(input_dim, output_dim, use_bias, stride)