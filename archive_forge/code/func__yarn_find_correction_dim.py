import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def _yarn_find_correction_dim(num_rotations: int, dim: int, base: float=10000, max_position_embeddings: int=2048) -> float:
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))