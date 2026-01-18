import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad):
    tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor