import shutil
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, _Stateful
class FSDPStrategy(ParallelStrategy, _Sharded):
    """Strategy for Fully Sharded Data Parallel provided by torch.distributed.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Fully Sharded Training shards the entire model across all available GPUs, allowing you to scale model
    size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
    at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
    to ZeRO-Stage 3.

    For more information check out
    `this blogpost <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api>`__.

    Defaults have been set and options have been exposed, but may require configuration
    based on your level of memory/speed efficiency. We suggest having a look at
    `this tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ for more information.

    Arguments:
        cpu_offload: See ``cpu_offload`` parameter in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
        mixed_precision: See ``mixed_precision`` parameter in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
        auto_wrap_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch.distributed.fsdp.FullyShardedDataParallel`. For convenience, this also accepts a set of the
            layer classes to wrap.
        activation_checkpointing: Deprecated. Use ``activation_checkpointing_policy``.
        activation_checkpointing_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch.distributed.fsdp.FullyShardedDataParallel` but used when selecting the modules for which you
            want to enable activation checkpointing. Enabling this can free up a significant amount of memory at the
            cost of speed since activations in these layers need to be recomputed during backpropagation. For
            convenience, this also accepts a set of the layer classes to wrap.
        sharding_strategy: Select whether to shard model parameters, gradients, optimizer states, or a combination of
            them. Available values are:

            - ``"FULL_SHARD"``: Shards model parameters, gradients, and optimizer states (default).
            - ``"SHARD_GRAD_OP"``: Shards gradients and optimizer states only. Model parameters get replicated.
            - ``"NO_SHARD"``: No sharding (identical to regular DDP).
            - ``"HYBRID_SHARD"``: Shards model parameters, gradients, and optimizer states within a single machine, but
              replicates across machines.

            Also accepts a :class:`torch.distributed.fsdp.ShardingStrategy` enum value.

        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"full"``: The full weights and optimizer states get assembled on rank 0 and saved to a single file.
            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with as many files as the world size.

        \\**kwargs: See available parameters in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    """

    def __init__(self, accelerator: Optional[Accelerator]=None, parallel_devices: Optional[List[torch.device]]=None, cluster_environment: Optional[ClusterEnvironment]=None, precision: Optional[Precision]=None, process_group_backend: Optional[str]=None, timeout: Optional[timedelta]=default_pg_timeout, cpu_offload: Union[bool, 'CPUOffload', None]=None, mixed_precision: Optional['MixedPrecision']=None, auto_wrap_policy: Optional['_POLICY']=None, activation_checkpointing: Optional[Union[Type[Module], List[Type[Module]]]]=None, activation_checkpointing_policy: Optional['_POLICY']=None, sharding_strategy: '_SHARDING_STRATEGY'='FULL_SHARD', state_dict_type: Literal['full', 'sharded']='sharded', **kwargs: Any) -> None:
        super().__init__(accelerator=accelerator, parallel_devices=parallel_devices, cluster_environment=cluster_environment, precision=precision)
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._backward_sync_control = _FSDPBackwardSyncControl()
        self._fsdp_kwargs = _auto_wrap_policy_kwargs(auto_wrap_policy, kwargs)
        if _TORCH_GREATER_EQUAL_2_0:
            self._fsdp_kwargs.setdefault('use_orig_params', True)
        self._activation_checkpointing_kwargs = _activation_checkpointing_kwargs(activation_checkpointing, activation_checkpointing_policy)
        self._state_dict_type = state_dict_type
        self.sharding_strategy = _init_sharding_strategy(sharding_strategy, self._fsdp_kwargs)
        self.cpu_offload = _init_cpu_offload(cpu_offload)
        self.mixed_precision = mixed_precision

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(f'The `{type(self).__name__}` does not use the `CheckpointIO` plugin interface.')

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: CheckpointIO) -> None:
        raise NotImplementedError(f'The `{type(self).__name__}` does not support setting a `CheckpointIO` plugin.')

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return {'num_replicas': self.num_nodes * self.num_processes, 'rank': self.global_rank}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    def mixed_precision_config(self) -> Optional['MixedPrecision']:
        if self.mixed_precision:
            return self.mixed_precision
        plugin = self.precision
        if isinstance(plugin, FSDPPrecision):
            return plugin.mixed_precision_config
        return None

    @property
    @override
    def precision(self) -> FSDPPrecision:
        plugin = self._precision
        if plugin is not None:
            assert isinstance(plugin, FSDPPrecision)
            return plugin
        return FSDPPrecision('32-true')

    @precision.setter
    @override
    def precision(self, precision: Optional[FSDPPrecision]) -> None:
        if precision is not None and (not isinstance(precision, FSDPPrecision)):
            raise TypeError(f'The FSDP strategy can only work with the `FSDPPrecision` plugin, found {precision}')
        self._precision = precision

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

    @override
    def setup_module_and_optimizers(self, module: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module and sets `use_orig_params=True` to keep the reference to the original parameters in the optimizer."""
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError(f'The `{type(self).__name__}` does not support the joint setup of module and optimizer(s). Please do it in this order: Create the model, call `setup_module`, create the optimizer, call `setup_optimizer`.')
        use_orig_params = self._fsdp_kwargs.get('use_orig_params')
        if use_orig_params is False:
            raise ValueError(f'You set `{type(self).__name__}(use_orig_params=False)` but this is not supported when setting the model and optimizer up jointly. Either set it to `True` or set the objects up in this order: Create the model, call `setup_module`, create the optimizer, call `setup_optimizer`.')
        module = self.setup_module(module)
        return (module, optimizers)

    @override
    def setup_module(self, module: Module) -> Module:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module."""
        from torch.distributed.fsdp import FullyShardedDataParallel
        if any((isinstance(mod, FullyShardedDataParallel) for mod in module.modules())):
            if _has_meta_device_parameters(module):
                rank_zero_warn('The model is already wrapped in `FSDP` but there are still parameters on the meta device.')
            if 'auto_wrap_policy' in self._fsdp_kwargs:
                rank_zero_warn('A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored.')
                del self._fsdp_kwargs['auto_wrap_policy']
        else:
            module = FullyShardedDataParallel(module=module, cpu_offload=self.cpu_offload, mixed_precision=self.mixed_precision_config, sharding_strategy=self.sharding_strategy, device_id=self.root_device.index, **self._fsdp_kwargs)
        _move_torchmetrics_to_device(module, self.root_device)
        _setup_activation_checkpointing(module, self._activation_checkpointing_kwargs)
        return module

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with FSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.

        """
        if self._fsdp_kwargs.get('use_orig_params'):
            return super().setup_optimizer(optimizer)
        if not _optimizer_has_flat_params(optimizer):
            raise ValueError('The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the optimizer after setting up the model.')
        return optimizer

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool]=None) -> ContextManager:
        precision_init_ctx = self.precision.module_init_context()
        module_sharded_ctx = self.module_sharded_context()
        empty_ctx = _EmptyInit(enabled=bool(empty_init))
        stack = ExitStack()
        if _TORCH_GREATER_EQUAL_2_1 and empty_init:
            stack.enter_context(torch.device('meta'))
        else:
            stack.enter_context(empty_ctx)
        stack.enter_context(precision_init_ctx)
        stack.enter_context(module_sharded_ctx)
        return stack

    @override
    def module_sharded_context(self) -> ContextManager:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap
        return enable_wrap(wrapper_cls=FullyShardedDataParallel, cpu_offload=self.cpu_offload, mixed_precision=self.mixed_precision_config, sharding_strategy=self.sharding_strategy, device_id=self.root_device.index, **self._fsdp_kwargs)

    @override
    def all_reduce(self, tensor: Tensor, group: Optional[Any]=None, reduce_op: Optional[Union[ReduceOp, str]]='mean') -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == 'nccl':
            torch.distributed.barrier(device_ids=[self.root_device.index])
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int=0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj
        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def clip_gradients_norm(self, module: Module, optimizer: Optimizer, max_norm: Union[float, int], norm_type: Union[float, int]=2.0, error_if_nonfinite: bool=True) -> Tensor:
        """Clip gradients by norm."""
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        if not isinstance(module, FullyShardedDataParallel):
            raise TypeError(f'Gradient clipping with FSDP is only possible if the module passed to `{self.__class__.__name__}.clip_gradients_norm` is wrapped in `FullyShardedDataParallel`. Got: {module.__class__.__name__}.')
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)

    @override
    def save_checkpoint(self, path: _PATH, state: Dict[str, Union[Module, Optimizer, Any]], storage_options: Optional[Any]=None, filter: Optional[Dict[str, Callable[[str, Any], bool]]]=None) -> None:
        """Save model, optimizer, and other state to a checkpoint on disk.

        If the state-dict-type is ``'full'``, the checkpoint will be written to a single file containing the weights,
        optimizer state and other metadata. If the state-dict-type is ``'sharded'``, the checkpoint gets saved as a
        directory containing one file per process, with model- and optimizer shards stored per file. Additionally, it
        creates a metadata file `meta.pt` with the rest of the user's state (only saved from rank 0).

        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError('Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0. Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`.')
        if storage_options is not None:
            raise TypeError('`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because `FSDPStrategy` does not use the `CheckpointIO`.')
        if filter is not None and self._state_dict_type == 'sharded':
            raise NotImplementedError("FSDP doesn't support loading sharded filtered checkpoints, so saving them is disabled.")
        path = Path(self.broadcast(path))
        if path.is_dir() and self._state_dict_type == 'full' and (not _is_sharded_checkpoint(path)):
            raise IsADirectoryError(f'The checkpoint path exists and is a directory: {path}')
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        modules = [module for module in state.values() if _has_fsdp_modules(module)]
        if len(modules) == 0:
            raise ValueError("Could not find a FSDP model in the provided checkpoint state. Please provide the model as part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure you set up the model (and optimizers if any) through the strategy before saving the checkpoint.")
        if len(modules) > 1:
            raise ValueError('Found multiple FSDP models in the given state. Saving checkpoints with FSDP is currently limited to a single model per checkpoint. To save multiple models, call the save method for each model separately with a different path.')
        module = modules[0]
        if self._state_dict_type == 'sharded':
            if path.is_file():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)
            state_dict_ctx = _get_sharded_state_dict_context(module)
            converted_state: Dict[str, Any] = {}
            metadata: Dict[str, Any] = {}
            with state_dict_ctx:
                for key, obj in state.items():
                    converted: Any
                    if isinstance(obj, Module):
                        converted = obj.state_dict()
                        target_dict = converted_state
                    elif isinstance(obj, Optimizer):
                        converted = FSDP.optim_state_dict(module, obj)
                        target_dict = converted_state
                    else:
                        converted = obj.state_dict() if isinstance(obj, _Stateful) else obj
                        target_dict = metadata
                    _apply_filter(key, filter or {}, converted, target_dict)
            _distributed_checkpoint_save(converted_state, path)
            if self.global_rank == 0:
                torch.save(metadata, path / _METADATA_FILENAME)
        elif self._state_dict_type == 'full':
            if _is_sharded_checkpoint(path):
                shutil.rmtree(path)
            state_dict_ctx = _get_full_state_dict_context(module, world_size=self.world_size)
            full_state: Dict[str, Any] = {}
            with state_dict_ctx:
                for key, obj in state.items():
                    if isinstance(obj, Module):
                        converted = obj.state_dict()
                    elif isinstance(obj, Optimizer):
                        converted = FSDP.optim_state_dict(module, obj)
                    else:
                        converted = obj.state_dict() if isinstance(obj, _Stateful) else obj
                    _apply_filter(key, filter or {}, converted, full_state)
            if self.global_rank == 0:
                torch.save(full_state, path)
        else:
            raise ValueError(f'Unknown state_dict_type: {self._state_dict_type}')

    @override
    def load_checkpoint(self, path: _PATH, state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]]=None, strict: bool=True) -> Dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects.

        The strategy currently only supports saving and loading sharded checkpoints which are stored in form of a
        directory of multiple files rather than a single file.

        """
        if not _TORCH_GREATER_EQUAL_2_0:
            raise NotImplementedError('Saving and loading checkpoints with the `FSDPStrategy` is not supported in PyTorch < 2.0. Please upgrade `torch` or file an issue: `https://github.com/Lightning-AI/lightning/issues`.')
        if not state:
            raise ValueError(f"Got FSDPStrategy.load_checkpoint(..., state={state!r}) but a state with at least  a model instance to reload is required. Pass it in like so: FSDPStrategy.load_checkpoint(..., state={{'model': model, ...}})")
        path = Path(self.broadcast(path))
        if isinstance(state, Module):
            _load_raw_module_state_from_path(path, module=state, world_size=self.world_size, strict=strict)
            return {}
        if isinstance(state, Optimizer):
            raise NotImplementedError('Loading a single optimizer object from a checkpoint is not supported yet with the FSDP strategy.')
        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import OptimStateKeyType
        modules = {key: module for key, module in state.items() if _has_fsdp_modules(module)}
        if len(modules) == 0:
            raise ValueError("Could not find a FSDP model in the provided checkpoint state. Please provide the model as part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure you set up the model (and optimizers if any) through the strategy before loading the checkpoint.")
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        if len(modules) > 1:
            raise ValueError('Found multiple FSDP models in the given state. Loading checkpoints with FSDP is currently limited to a single model per checkpoint. To load multiple models, call the load method for each model separately with a different path.')
        module_key, module = list(modules.items())[0]
        if _is_sharded_checkpoint(path):
            state_dict_ctx = _get_sharded_state_dict_context(module)
            with state_dict_ctx:
                module_state = {module_key: module.state_dict()}
                _distributed_checkpoint_load(module_state, path)
                module.load_state_dict(module_state[module_key], strict=strict)
                if optimizers:
                    from torch.distributed.checkpoint import FileSystemReader
                    reader = FileSystemReader(path=path)
                    for optim_key, optim in optimizers.items():
                        optim_state = load_sharded_optimizer_state_dict(model_state_dict=module_state[module_key], optimizer_key=optim_key, storage_reader=reader)
                        flattened_osd = FSDP.optim_state_dict_to_load(optim_state_dict=optim_state[optim_key], model=module, optim=optim)
                        optim.load_state_dict(flattened_osd)
            metadata = torch.load(path / _METADATA_FILENAME)
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, metadata.keys(), strict=strict)
            for key in requested_metadata_keys:
                if key not in metadata:
                    continue
                state[key] = metadata.pop(key)
            return metadata
        if _is_full_checkpoint(path):
            checkpoint = _lazy_load(path) if _TORCH_GREATER_EQUAL_2_0 else torch.load(path, map_location='cpu')
            _load_raw_module_state(checkpoint.pop(module_key), module=module, world_size=self.world_size, strict=strict)
            if isinstance(state, Module):
                return {}
            if _TORCH_GREATER_EQUAL_2_0:
                checkpoint = _materialize_tensors(checkpoint)
            for optim_key, optim in optimizers.items():
                with _get_full_state_dict_context(module, world_size=self.world_size, rank0_only=False):
                    temp_state_dict = checkpoint.pop(optim_key)
                    if isinstance(list(temp_state_dict['state'].keys())[0], int):
                        temp_state_dict = FSDP.rekey_optim_state_dict(temp_state_dict, OptimStateKeyType.PARAM_NAME, module)
                    optim_state_dict = FSDP.optim_state_dict_to_load(optim_state_dict=temp_state_dict, model=module, optim=optim)
                    optim.load_state_dict(optim_state_dict)
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, checkpoint.keys(), strict=strict)
            _move_state_into(source=checkpoint, destination=state, keys=requested_metadata_keys)
            return checkpoint
        raise ValueError(f'The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a directory with FSDP checkpoint shards, or a single file with a full checkpoint.')

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not torch.distributed.is_available():
            return
        strategy_registry.register('fsdp', cls, description='Fully Sharded Data Parallel (FSDP) training')
        strategy_registry.register('fsdp_cpu_offload', cls, description='Fully Sharded Data Parallel (FSDP) training with Full Sharding and CPU Offloading', cpu_offload=True)

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank