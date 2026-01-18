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
class _FeatureExtractor(ABC):

    @abstractmethod
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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