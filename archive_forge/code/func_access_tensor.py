from __future__ import annotations
import dataclasses
from . import torch_wrapper
@property
def access_tensor(self) -> torch.Tensor:
    return torch.tensor(self.storage, dtype=self.dtype, device=self.storage.device)