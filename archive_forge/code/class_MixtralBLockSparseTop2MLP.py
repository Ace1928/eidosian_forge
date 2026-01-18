import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from ...utils.import_utils import is_torch_fx_available
from .configuration_mixtral import MixtralConfig
class MixtralBLockSparseTop2MLP(MixtralBlockSparseTop2MLP):

    def __init__(self, *args, **kwargs):
        logger.warning_once('MixtralBLockSparseTop2MLP is deprecated by MixtralBlockSparseTop2MLP and will be removed in v4.40.')
        super().__init__(*args, **kwargs)