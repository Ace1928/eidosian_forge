import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
class _FSDPState(_State):

    def __init__(self) -> None:
        self._ignored_modules: Set[nn.Module] = set()
        self._ignored_params: Set[nn.Parameter] = set()
        self._ignored_buffer_names: Set[str] = set()
        self.process_group: Optional[dist.ProcessGroup] = None
        self.rank: int = -1
        self.world_size: int = -1
        self._device_mesh: Optional[DeviceMesh] = None
        self.sharding_strategy = ShardingStrategy.FULL_SHARD
        self._use_orig_params: bool = False
        self.training_state = TrainingState.IDLE
        self._unshard_params_ctx: Dict[nn.Module, Generator] = {}
        self._state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT
        self._state_dict_config: StateDictConfig = FullStateDictConfig()
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()
        self._is_root: Optional[bool] = None
        self._handle: Optional[flat_param_file.FlatParamHandle] = None
        self._fully_sharded_module_to_handle: Dict[nn.Module, Optional[flat_param_file.FlatParamHandle]] = {}
        self.compute_device: Optional[torch.device] = None
        self._gradient_predivide_factor: int = 0
        self._gradient_postdivide_factor: int = 0
        self._comm_hook: Optional[Callable] = None
        self._comm_hook_state: Optional[Any] = None
        self._device_handle: _FSDPDeviceHandle = _UninitializedDeviceHandle()
        self._all_fsdp_states: List[_FSDPState] = []
        self._all_handles: List[flat_param_file.FlatParamHandle] = []
        self._fsdp_extension: Optional[FSDPExtensions] = None