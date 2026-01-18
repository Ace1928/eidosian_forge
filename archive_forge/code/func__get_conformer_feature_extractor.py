from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def _get_conformer_feature_extractor(input_dim: int, output_dim: int, stride: int) -> FeatureEncoder:
    """Construct Feature Extractor

    Args:
        input_dim (int): Input dimension of features.
        output_dim (int): Output dimension after feature extraction.
        stride (int): Stride used in Time Reduction layer of feature extractor.

    Returns:
        FeatureEncoder: The resulting feature extraction.
    """
    return FeatureEncoder(input_dim, output_dim, stride)