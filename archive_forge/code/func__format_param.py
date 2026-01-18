import itertools
import math
from copy import deepcopy
import warnings
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
@staticmethod
def _format_param(optimizer, swa_lrs):
    if isinstance(swa_lrs, (list, tuple)):
        if len(swa_lrs) != len(optimizer.param_groups):
            raise ValueError(f'swa_lr must have the same length as optimizer.param_groups: swa_lr has {len(swa_lrs)}, optimizer.param_groups has {len(optimizer.param_groups)}')
        return swa_lrs
    else:
        return [swa_lrs] * len(optimizer.param_groups)