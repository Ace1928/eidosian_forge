import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def _is_complete(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() == storage_ptr(tensor) and tensor.nelement() * _SIZE[tensor.dtype] == storage_size(tensor)