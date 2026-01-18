import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
class _ModuleFeatureExtractor(torch.nn.Module, _FeatureExtractor):
    """``torch.nn.Module``-based feature extraction pipeline.

    Args:
        pipeline (torch.nn.Module): module that implements feature extraction logic.
    """

    def __init__(self, pipeline: torch.nn.Module) -> None:
        super().__init__()
        self.pipeline = pipeline

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates features and length output from the given input tensor.

        Args:
            input (torch.Tensor): input tensor.

        Returns:
            (torch.Tensor, torch.Tensor):
            torch.Tensor:
                Features, with shape `(length, *)`.
            torch.Tensor:
                Length, with shape `(1,)`.
        """
        features = self.pipeline(input)
        length = torch.tensor([features.shape[0]])
        return (features, length)