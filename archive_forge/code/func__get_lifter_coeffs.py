import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def _get_lifter_coeffs(num_ceps: int, cepstral_lifter: float) -> Tensor:
    i = torch.arange(num_ceps)
    return 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * i / cepstral_lifter)