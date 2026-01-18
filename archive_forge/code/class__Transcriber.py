from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
class _Transcriber(ABC):

    @abstractmethod
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        pass