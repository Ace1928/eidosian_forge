import logging
import os
from collections import Counter
from typing import Dict, List, Literal, Optional, Union
import torch
from lightning_fabric.connector import _PRECISION_INPUT, _PRECISION_INPUT_STR, _convert_precision_to_unified_args
from lightning_fabric.plugins.environments import (
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.plugins import (
from pytorch_lightning.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from pytorch_lightning.strategies import (
from pytorch_lightning.strategies.ddp import _DDP_FORK_ALIASES
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import (
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _check_config_and_set_final_flags(self, strategy: Union[str, Strategy], accelerator: Union[str, Accelerator], precision: Optional[_PRECISION_INPUT], plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]], sync_batchnorm: bool) -> None:
    """This method checks:

        1. strategy: whether the strategy name is valid, and sets the internal flags if it is.
        2. accelerator: if the value of the accelerator argument is a type of accelerator (instance or string),
            set self._accelerator_flag accordingly.
        3. precision: The final value of the precision flag may be determined either by the precision argument or
            by a plugin instance.
        4. plugins: The list of plugins may contain a Precision plugin, CheckpointIO, ClusterEnvironment and others.
            Additionally, other flags such as `precision` or `sync_batchnorm` can populate the list with the
            corresponding plugin instances.

        """
    if plugins is not None:
        plugins = [plugins] if not isinstance(plugins, list) else plugins
    if isinstance(strategy, str):
        strategy = strategy.lower()
    self._strategy_flag = strategy
    if strategy == 'colossalai' and (not _LIGHTNING_COLOSSALAI_AVAILABLE):
        raise ModuleNotFoundError(str(_LIGHTNING_COLOSSALAI_AVAILABLE))
    if strategy == 'bagua' and (not _LIGHTNING_BAGUA_AVAILABLE):
        raise ModuleNotFoundError(str(_LIGHTNING_BAGUA_AVAILABLE))
    if strategy != 'auto' and strategy not in self._registered_strategies and (not isinstance(strategy, Strategy)):
        raise ValueError(f'You selected an invalid strategy name: `strategy={strategy!r}`. It must be either a string or an instance of `pytorch_lightning.strategies.Strategy`. Example choices: auto, ddp, ddp_spawn, deepspeed, ... Find a complete list of options in our documentation at https://lightning.ai')
    if accelerator not in self._accelerator_types and accelerator not in ('auto', 'gpu') and (not isinstance(accelerator, Accelerator)):
        raise ValueError(f'You selected an invalid accelerator name: `accelerator={accelerator!r}`. Available names are: auto, {', '.join(self._accelerator_types)}.')
    is_ddp_str = isinstance(strategy, str) and 'ddp' in strategy
    is_deepspeed_str = isinstance(strategy, str) and 'deepspeed' in strategy
    is_parallel_strategy = isinstance(strategy, ParallelStrategy) or is_ddp_str or is_deepspeed_str
    is_mps_accelerator = MPSAccelerator.is_available() and (accelerator in ('mps', 'auto', 'gpu', None) or isinstance(accelerator, MPSAccelerator))
    if is_mps_accelerator and is_parallel_strategy:
        raise ValueError(f"You set `strategy={strategy}` but strategies from the DDP family are not supported on the MPS accelerator. Either explicitly set `accelerator='cpu'` or change the strategy.")
    self._accelerator_flag = accelerator
    precision_flag = _convert_precision_to_unified_args(precision)
    if plugins:
        plugins_flags_types: Dict[str, int] = Counter()
        for plugin in plugins:
            if isinstance(plugin, Precision):
                self._precision_plugin_flag = plugin
                plugins_flags_types[Precision.__name__] += 1
            elif isinstance(plugin, CheckpointIO):
                self.checkpoint_io = plugin
                plugins_flags_types[CheckpointIO.__name__] += 1
            elif isinstance(plugin, ClusterEnvironment):
                self._cluster_environment_flag = plugin
                plugins_flags_types[ClusterEnvironment.__name__] += 1
            elif isinstance(plugin, LayerSync):
                if sync_batchnorm and (not isinstance(plugin, TorchSyncBatchNorm)):
                    raise MisconfigurationException(f'You set `Trainer(sync_batchnorm=True)` and provided a `{plugin.__class__.__name__}` plugin, but this is not allowed. Choose one or the other.')
                self._layer_sync = plugin
                plugins_flags_types[TorchSyncBatchNorm.__name__] += 1
            else:
                raise MisconfigurationException(f'Found invalid type for plugin {plugin}. Expected one of: Precision, CheckpointIO, ClusterEnviroment, or LayerSync.')
        duplicated_plugin_key = [k for k, v in plugins_flags_types.items() if v > 1]
        if duplicated_plugin_key:
            raise MisconfigurationException(f'Received multiple values for {', '.join(duplicated_plugin_key)} flags in `plugins`. Expected one value for each type at most.')
        if plugins_flags_types.get(Precision.__name__) and precision_flag is not None:
            raise ValueError(f'Received both `precision={precision_flag}` and `plugins={self._precision_plugin_flag}`. Choose one.')
    self._precision_flag = '32-true' if precision_flag is None else precision_flag
    if self._strategy_flag and isinstance(self._strategy_flag, Strategy):
        if self._strategy_flag._accelerator:
            if self._accelerator_flag != 'auto':
                raise MisconfigurationException('accelerator set through both strategy class and accelerator flag, choose one')
            self._accelerator_flag = self._strategy_flag._accelerator
        if self._strategy_flag._precision_plugin:
            if self._precision_plugin_flag:
                raise MisconfigurationException('precision set through both strategy class and plugins, choose one')
            self._precision_plugin_flag = self._strategy_flag._precision_plugin
        if self._strategy_flag._checkpoint_io:
            if self.checkpoint_io:
                raise MisconfigurationException('checkpoint_io set through both strategy class and plugins, choose one')
            self.checkpoint_io = self._strategy_flag._checkpoint_io
        if getattr(self._strategy_flag, 'cluster_environment', None):
            if self._cluster_environment_flag:
                raise MisconfigurationException('cluster_environment set through both strategy class and plugins, choose one')
            self._cluster_environment_flag = getattr(self._strategy_flag, 'cluster_environment')
        if hasattr(self._strategy_flag, 'parallel_devices') and self._strategy_flag.parallel_devices:
            if self._strategy_flag.parallel_devices[0].type == 'cpu':
                if self._accelerator_flag and self._accelerator_flag not in ('auto', 'cpu'):
                    raise MisconfigurationException(f'CPU parallel_devices set through {self._strategy_flag.__class__.__name__} class, but accelerator set to {self._accelerator_flag}, please choose one device type')
                self._accelerator_flag = 'cpu'
            if self._strategy_flag.parallel_devices[0].type == 'cuda':
                if self._accelerator_flag and self._accelerator_flag not in ('auto', 'cuda', 'gpu'):
                    raise MisconfigurationException(f'GPU parallel_devices set through {self._strategy_flag.__class__.__name__} class, but accelerator set to {self._accelerator_flag}, please choose one device type')
                self._accelerator_flag = 'cuda'
            self._parallel_devices = self._strategy_flag.parallel_devices