import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
class _TorchAccelerator(Accelerator):
    """A utility that implements methods to accelerate PyTorch training.

    Arguments:
        amp: If true, perform training with automatic mixed precision.
            Otherwise, use full precision.
    """

    def __init__(self, amp: bool=False):
        self.amp_is_enabled = amp
        self.scaler = GradScaler() if amp else None
        self._seed = None

    def prepare_model(self, model: torch.nn.Module, move_to_device: bool=True, parallel_strategy: Optional[str]='ddp', parallel_strategy_kwargs: Optional[Dict[str, Any]]=None) -> torch.nn.Module:
        """Prepares the model for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            model (torch.nn.Module): A torch model to prepare.
            move_to_device: Whether to move the model to the correct
                device. If set to False, the model needs to manually be moved
                to the correct device.
            parallel_strategy ("ddp", "fsdp", or None): Whether to wrap models
                in ``DistributedDataParallel``, ``FullyShardedDataParallel`` (
                Experimental), or neither.
            parallel_strategy_kwargs (Dict[str, Any]): Args to pass into
                ``DistributedDataParallel`` or ``FullyShardedDataParallel``
                initialization if ``parallel_strategy`` is set to "ddp"
                or "fsdp", respectively.
        """
        parallel_strategy_kwargs = parallel_strategy_kwargs or {}
        rank = session.get_local_rank()
        if isinstance(move_to_device, torch.device):
            device = move_to_device
        else:
            device = get_device()
            if isinstance(device, list):
                device = device[0]
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        if move_to_device:
            if rank == 0:
                logger.info(f'Moving model to device: {device}')
            else:
                logger.debug(f'Moving model to device: {device}')
            model = model.to(device)

        def model_get_state(self):
            if hasattr(self, '_original_get_state'):
                state = self._original_get_state()
                state['__getstate__'] = state['_original_get_state']
                del state['_original_get_state']
            else:
                state = self.__dict__.copy()
                del state['__getstate__']
            state['forward'] = state['_unwrapped_forward']
            del state['_unwrapped_forward']
            return state
        if self.amp_is_enabled:
            model._unwrapped_forward = model.forward
            model.forward = autocast()(model.forward)
            if hasattr(model, '__getstate__'):
                model._original_get_state = model.__getstate__
            model.__getstate__ = types.MethodType(model_get_state, model)
        world_size = session.get_world_size()
        if parallel_strategy and world_size > 1:
            if parallel_strategy == 'ddp':
                DataParallel = DistributedDataParallel
                if torch.cuda.is_available():
                    parallel_strategy_kwargs = {'device_ids': [device], 'output_device': device, **parallel_strategy_kwargs}
            else:
                if not torch.cuda.is_available():
                    raise RuntimeError('FSDP is only available with GPU-enabled training. Set `use_gpu=True` in your Trainer to train with GPUs.')
                DataParallel = FullyShardedDataParallel
            if rank == 0:
                logger.info(f'Wrapping provided model in {DataParallel.__name__}.')
            else:
                logger.debug(f'Wrapping provided model in {DataParallel.__name__}.')
            model = DataParallel(model, **parallel_strategy_kwargs)
        return model

    def prepare_data_loader(self, data_loader: torch.utils.data.DataLoader, add_dist_sampler: bool=True, move_to_device: bool=True, auto_transfer: bool=False) -> torch.utils.data.DataLoader:
        """Prepares DataLoader for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to
                prepare.
            add_dist_sampler: Whether to add a DistributedSampler to
                the provided DataLoader.
            move_to_device: If set, automatically move the data
                returned by the data loader to the correct device.
            auto_transfer: (Experimental) If set and device is GPU, another CUDA stream
                is created to automatically copy data from host (CPU) memory
                to device (GPU) memory (the default CUDA stream still runs the
                training procedure). If device is CPU, it will be disabled
                regardless of the setting. This configuration will be ignored
                if ``move_to_device`` is False.
        """
        world_size = session.get_world_size()
        world_rank = session.get_world_rank()
        if world_size > 1 and (not isinstance(data_loader.sampler, DistributedSampler)) and (not (hasattr(data_loader, 'dataset') and isinstance(data_loader.dataset, IterableDataset))) and add_dist_sampler:

            def with_sampler(loader):
                shuffle = not isinstance(loader.sampler, SequentialSampler)

                def seeded_worker_init_fn(worker_init_fn: Optional[Callable[[int], None]]):

                    def wrapper(worker_id: int):
                        worker_seed = torch.initial_seed() % 2 ** 32
                        np.random.seed(worker_seed)
                        random.seed(worker_seed)
                        if worker_init_fn:
                            worker_init_fn(worker_id)
                    return wrapper
                worker_init_fn: Optional[Callable[[int], None]] = loader.worker_init_fn
                generator: Optional[torch.Generator] = loader.generator
                if self._seed is not None:
                    worker_init_fn = seeded_worker_init_fn(worker_init_fn)
                    generator = torch.Generator()
                    generator.manual_seed(self._seed)
                using_default_sampler = isinstance(loader.sampler, (SequentialSampler, RandomSampler))
                if not using_default_sampler and world_rank == 0:
                    logger.warn(f'The {loader.sampler.__class__.__name__} will be overwritten with a DistributedSampler. You can disable this by setting `with_sampler` to False in `prepare_data_loader`.')
                data_loader_args = {'dataset': loader.dataset, 'batch_size': loader.batch_size, 'shuffle': False, 'num_workers': loader.num_workers, 'collate_fn': loader.collate_fn, 'pin_memory': loader.pin_memory, 'drop_last': loader.drop_last, 'timeout': loader.timeout, 'worker_init_fn': worker_init_fn, 'generator': generator, 'sampler': DistributedSampler(loader.dataset, shuffle=shuffle)}
                return DataLoader(**data_loader_args)
            data_loader = with_sampler(data_loader)
        if move_to_device:
            device = get_device()
            data_loader = _WrappedDataLoader(data_loader, device, auto_transfer)
        return data_loader

    def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Wraps optimizer to support automatic mixed precision.

        Args:
            optimizer (torch.optim.Optimizer): The DataLoader to prepare.

        Returns:
            A wrapped optimizer.
        """
        return _WrappedOptimizer(optimizer, scaler=self.scaler)

    def backward(self, tensor: torch.Tensor) -> None:
        """Computes the gradient of the specified tensor w.r.t. graph leaves.

        Args:
            tensor (torch.Tensor): Tensor of which the derivative will be computed.
        """
        if self.amp_is_enabled:
            self.scaler.scale(tensor).backward()
        else:
            tensor.backward()

    def enable_reproducibility(self, seed: int=0) -> None:
        """Limits sources of nondeterministic behavior."""
        self._seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'