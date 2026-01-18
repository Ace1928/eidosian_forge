import contextlib
import copy
import functools
import math
import traceback
import warnings
from contextlib import contextmanager
from enum import auto, Enum
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._init_utils import (
from torch.distributed.fsdp._runtime_utils import (
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp.api import (
from torch.distributed.utils import _p_assert
from ._flat_param import FlatParameter
from ._optim_utils import (
from ._state_dict_utils import _register_all_state_dict_hooks
from ._unshard_param_utils import (
from .wrap import CustomPolicy, ModuleWrapPolicy
def _assert_state(self, state: Union[TrainingState, List[TrainingState]]) -> None:
    """Assert we are in the given state."""
    if isinstance(state, TrainingState):
        state = [state]
    if self.training_state not in state:
        msg = f'expected to be in states {state} but current state is {self.training_state}'
        if self.rank == 0:
            print(f'Asserting FSDP instance is: {self}')
            print(f'ERROR: {msg}')
            traceback.print_stack()
        raise ValueError(msg)