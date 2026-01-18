from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional import TokenSpan
def _align_emission_and_tokens(emission: Tensor, tokens: List[int], blank: int=0):
    device = emission.device
    emission = emission.unsqueeze(0)
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    aligned_tokens, scores = F.forced_align(emission, targets, blank=blank)
    scores = scores.exp()
    aligned_tokens, scores = (aligned_tokens[0], scores[0])
    return (aligned_tokens, scores)