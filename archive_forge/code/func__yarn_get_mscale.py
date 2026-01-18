import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def _yarn_get_mscale(scale: float=1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0