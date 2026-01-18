from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
class NegativeSampler(Module):
    """Applies preprocessing to input and then computes negative sampling.

    Args:
        preprocessor (nn.Module): Transforms input tensor prior to negative sampling.
        num_negatives (int): Number of negative examples to sample.
        cross_sample_negatives (int): Number of negative examples to cross sample.
    """

    def __init__(self, preprocessor: Module, num_negatives: int, cross_sample_negatives: int):
        super().__init__()
        self.preprocessor = preprocessor
        self.num_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            input (Tensor): Tensor of dimension `(B, T, D)`.

        Returns:
            (Tensor, Tensor, Optional[Tensor]):
            Tensor
                The input tensor after preprocessing, prior to being sampled.
            Tensor
                The negative samples.
            Tensor
                The indices of the negative samples.
        """
        preprocessed = self.preprocessor(input)
        negs, neg_idxs = _sample_negatives(preprocessed, self.num_negatives, self.cross_sample_negatives)
        return (preprocessed, negs, neg_idxs)