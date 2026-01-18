from typing import Callable, Iterable, Optional, Union
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import contract
from torch.distributed._composable_state import _get_module_state, _insert_module_state
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp._state_dict_utils import _register_all_state_dict_hooks
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
from torch.distributed.fsdp.wrap import _Policy

    Applies ``FullyShardedDataParallel` (FSDP) semantics to ``module``.
    