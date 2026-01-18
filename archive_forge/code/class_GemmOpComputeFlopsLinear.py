import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
class GemmOpComputeFlopsLinear(GemmOpComputeFlops):

    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        return (prod(inputs[0].shape[:-1]), inputs[1].shape[0], inputs[0].shape[-1])