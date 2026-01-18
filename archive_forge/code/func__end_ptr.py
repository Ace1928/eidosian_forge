import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _SIZE[tensor.dtype]
    else:
        stop = tensor.data_ptr()
    return stop