import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms
class FSDPTest(MultiProcessTestCase):

    def setUp(self):
        super().setUp()
        os.environ['TORCH_NCCL_DESYNC_DEBUG'] = '0'
        self._spawn_processes()

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 8) if torch.cuda.is_available() else 4

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def init_method(self):
        return f'{FILE_SCHEMA}{self.file_name}'

    def _check_cpu_offload(self, fsdp_model, cpu_offload):
        self.assertEqual(cpu_offload, fsdp_model.cpu_offload)

    def _check_backward_prefetch(self, fsdp_model, backward_prefetch):
        self.assertEqual(backward_prefetch, fsdp_model.backward_prefetch)

    def _check_forward_prefetch(self, fsdp_model, forward_prefetch):
        self.assertEqual(forward_prefetch, fsdp_model.forward_prefetch)

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        print(f'dist init r={self.rank}, world={self.world_size}')
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        try:
            dist.init_process_group(init_method=self.init_method, backend=backend, world_size=int(self.world_size), rank=self.rank)
        except RuntimeError as e:
            if 'recompile' in e.args[0]:
                sys.exit(TEST_SKIPS['backend_unavailable'].exit_code)
            raise
        if torch.cuda.is_available() and torch.cuda.device_count():
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
        dist.barrier()
        self.run_test(test_name, pipe)
        dist.barrier()
        dist.destroy_process_group()

    def _train_for_several_steps(self, model: nn.Module, num_steps: int, autocast: bool, lr: float=0.01, fsdp_cpu_offload: Optional[CPUOffload]=None, save_model: bool=False, mixed_precision: Optional[MixedPrecision]=None, enable_sharded_grad_scaler: bool=False, use_pure_fp16: bool=False, sharded_grad_scaler_kwargs: Optional[Dict[str, Any]]=None):
        cpu_offload_params = fsdp_cpu_offload and fsdp_cpu_offload.offload_params
        model_device = next(model.parameters()).device
        if sharded_grad_scaler_kwargs is None:
            sharded_grad_scaler_kwargs = {}
        sharded_grad_scaler = ShardedGradScaler(enabled=enable_sharded_grad_scaler, **sharded_grad_scaler_kwargs)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                input = model.module.get_input(torch.device('cuda'))
                if use_pure_fp16 or (mixed_precision and (not isinstance(model, FSDP))):
                    if isinstance(input, torch.Tensor):
                        input = input.half()
                    else:
                        input = tuple((x.half() for x in input))
                output = model(*input)
                if cpu_offload_params and isinstance(model, FSDP) and (model.sharding_strategy not in NO_RESHARD_AFTER_FORWARD_STRATEGIES):
                    for p in model.parameters():
                        self.assertEqual(p.device, torch.device('cpu'))
                loss = model.module.get_loss(input, output).to(model_device)
            loss = sharded_grad_scaler.scale(loss)
            if not mixed_precision and (not use_pure_fp16):
                assert loss.dtype == torch.float32, 'loss data type should be float32, as the original                     parameter data type is float32.'
            elif use_pure_fp16:
                self.assertEqual(loss.dtype, torch.float16)
            elif isinstance(model, FSDP):
                self.assertEqual(loss.dtype, mixed_precision.param_dtype)
            else:
                self.assertEqual(loss.dtype, torch.float32)
            model.module.run_backward(loss)
            if cpu_offload_params and isinstance(model, FSDP):
                for p in model.parameters():
                    self.assertEqual(p.device, torch.device('cpu'))
            sharded_grad_scaler.step(optim)
            sharded_grad_scaler.update()
            if save_model:
                state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                _zero_model(model)
                model.load_state_dict(state_dict)
        if isinstance(model, FSDP):
            model._assert_state(TrainingState.IDLE)
        return loss.detach()

    def _test_fsdp_parity(self, model_class: Type[FSDPTestModel], fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, ref_init_fn: Optional[Callable]=None, num_iters: int=2, save_model: bool=True, cpu_offload: CPUOffload=CPUOffload(), backward_prefetch: Optional[BackwardPrefetch]=None, sharding_strategy: Optional[ShardingStrategy]=None, mixed_precision: Optional[MixedPrecision]=None, forward_prefetch: bool=False, use_orig_params: bool=False, enable_sharded_grad_scaler: bool=False, use_pure_fp16: bool=False, init_kwargs: Optional[Dict[str, Any]]=None, sharded_grad_scaler_kwargs: Optional[Dict[str, Any]]=None, **fsdp_kwargs):
        """
        Tests FSDP training against a reference, which defaults to DDP but
        may be customized with ``ref_init_fn``.

        Args:
            model_class (Type[FSDPTestModel]): A model class that inherits from
                ``FSDPTestModel``, which defines the expected interface.
            fsdp_init_mode (FSDPInitMode): The mode to initialize the
                FSDP-wrapped model. This should not be ``NO_FSDP``.
            ref_init_fn (Optional[Callable]): A callable to invoke that wraps a
                non-wrapped model to construct the reference model, where this
                wrapper should provide data parallel semantics. If ``None``,
                then the callable defaults to the DDP constructor.
        """
        assert fsdp_init_mode != FSDPInitMode.NO_FSDP, 'Expects an FSDP init mode that wraps with FSDP'
        if init_kwargs is None:
            init_kwargs = {}
        lr = 0.01
        rank = self.process_group.rank()
        model = model_class.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, deterministic=True, **init_kwargs)
        if ref_init_fn is None:
            ref_model = DDP(model, device_ids=[rank], output_device=rank)
        else:
            ref_model = ref_init_fn(model)
        if use_pure_fp16:
            ref_model = ref_model.half()
        ref_loss = self._train_for_several_steps(ref_model, num_iters, autocast=mixed_precision is not None, lr=lr, fsdp_cpu_offload=cpu_offload, mixed_precision=mixed_precision, enable_sharded_grad_scaler=enable_sharded_grad_scaler, use_pure_fp16=use_pure_fp16, sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs)
        ddp_params = list(ref_model.parameters())
        fsdp_kwargs.update({'cpu_offload': cpu_offload, 'backward_prefetch': backward_prefetch, 'sharding_strategy': sharding_strategy, 'mixed_precision': mixed_precision, 'forward_prefetch': forward_prefetch, 'use_orig_params': use_orig_params})
        try:
            fsdp_model = model_class.init(self.process_group, fsdp_init_mode, cuda_init_mode, fsdp_kwargs, deterministic=True, **init_kwargs)
        except Exception as e:
            raise ValueError(f'Initializing {model_class} raised error {str(e)}') from e
        if not isinstance(fsdp_model, FSDP):
            fsdp_model = FSDP(fsdp_model, self.process_group, **fsdp_kwargs)
        if use_pure_fp16:
            fsdp_model = fsdp_model.half()
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            fsdp_model = fsdp_model.cuda()
        offload_params = cpu_offload is not None and cpu_offload.offload_params
        expects_device_error = offload_params and cuda_init_mode == CUDAInitMode.CUDA_AFTER
        expects_cpu_device = offload_params and cuda_init_mode != CUDAInitMode.CUDA_AFTER
        if expects_cpu_device:
            cpu_device = torch.device('cpu')
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
        context = self.assertRaisesRegex(RuntimeError, 'An FSDP-managed module with parameter CPU offloading enabled has parameters on cuda') if expects_device_error else nullcontext()
        with context:
            fsdp_loss = self._train_for_several_steps(fsdp_model, num_iters, autocast=False, lr=lr, fsdp_cpu_offload=cpu_offload, save_model=save_model, mixed_precision=mixed_precision, enable_sharded_grad_scaler=enable_sharded_grad_scaler, use_pure_fp16=use_pure_fp16, sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs)
        if expects_device_error:
            return
        if offload_params:
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
            fsdp_loss = fsdp_loss.cuda()
        fsdp_unsharded_params = get_full_params(fsdp_model)
        torch.testing.assert_close(ref_loss, fsdp_loss, check_dtype=False)
        if mixed_precision is None and (not use_pure_fp16):
            self.assertEqual(ddp_params, fsdp_unsharded_params, exact_device=True, msg='FSDP did not match DDP')