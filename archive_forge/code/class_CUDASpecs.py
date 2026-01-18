import dataclasses
from typing import List, Optional, Tuple
import torch
@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: Tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: Tuple[int, int]

    @property
    def has_cublaslt(self) -> bool:
        return self.highest_compute_capability >= (7, 5)