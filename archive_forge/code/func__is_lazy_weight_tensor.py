import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
def _is_lazy_weight_tensor(p: Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter
    if isinstance(p, UninitializedParameter):
        warning_cache.warn('A layer with UninitializedParameter was found. Thus, the total number of parameters detected may be inaccurate.')
        return True
    return False