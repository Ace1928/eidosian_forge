import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def _wrap_model(self, model, training=True, dataloader=None):
    if self.args.use_ipex:
        dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
        model = self.ipex_optimize_model(model, training, dtype=dtype)
    if is_sagemaker_mp_enabled():
        if isinstance(self.model_wrapped, smp.model.DistributedModel):
            return self.model_wrapped
        return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)
    if unwrap_model(model) is not model:
        return model
    if self.use_apex and training:
        model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)
    if self.args.n_gpu > 1 and (not getattr(model, 'is_loaded_in_8bit', False)):
        model = nn.DataParallel(model)
    if self.args.jit_mode_eval:
        start_time = time.time()
        model = self.torch_jit_model_eval(model, dataloader, training)
        self.jit_compilation_time = round(time.time() - start_time, 4)
    if not training:
        return model
    if self.is_fsdp_xla_enabled:
        try:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
            from torch_xla.distributed.fsdp import checkpoint_module
            from torch_xla.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
            if self.is_fsdp_xla_v2_enabled:
                from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
        except ImportError:
            raise ImportError('Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.')
        auto_wrap_policy = None
        auto_wrapper_callable = None
        default_transformer_cls_names_to_wrap = getattr(model, '_no_split_modules', None)
        fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get('transformer_layer_cls_to_wrap', default_transformer_cls_names_to_wrap)
        if self.args.fsdp_config['min_num_params'] > 0:
            auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config['min_num_params'])
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception('Could not find the transformer layer class to wrap in the model.')
                else:
                    transformer_cls_to_wrap.add(transformer_cls)
            auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap)
        fsdp_kwargs = self.args.xla_fsdp_config
        if self.args.fsdp_config['xla_fsdp_grad_ckpt']:

            def auto_wrapper_callable(m, *args, **kwargs):
                target_cls = FSDP if not self.is_fsdp_xla_v2_enabled else FSDPv2
                return target_cls(checkpoint_module(m), *args, **kwargs)
        if self.is_fsdp_xla_v2_enabled:

            def shard_output(output, mesh):
                from .modeling_outputs import CausalLMOutputWithPast
                real_output = None
                if isinstance(output, torch.Tensor):
                    real_output = output
                elif isinstance(output, tuple):
                    real_output = output[0]
                elif isinstance(output, CausalLMOutputWithPast):
                    real_output = output.logits
                if real_output is None:
                    raise ValueError("Something went wrong, the output of the model shouldn't be `None`")
                xs.mark_sharding(real_output, mesh, ('fsdp', None, None))
            self.model = model = FSDPv2(model, shard_output=shard_output, auto_wrap_policy=auto_wrap_policy, auto_wrapper_callable=auto_wrapper_callable)
        else:
            self.model = model = FSDP(model, auto_wrap_policy=auto_wrap_policy, auto_wrapper_callable=auto_wrapper_callable, **fsdp_kwargs)

        def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
            loss = optimizer.step(**optimizer_args)
            if barrier:
                xm.mark_step()
            return loss
        xm.optimizer_step = patched_optimizer_step
    elif is_sagemaker_dp_enabled():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.getenv('SMDATAPARALLEL_LOCAL_RANK'))])
    elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
        if is_torch_neuroncore_available():
            return model
        kwargs = {}
        if self.args.ddp_find_unused_parameters is not None:
            kwargs['find_unused_parameters'] = self.args.ddp_find_unused_parameters
        elif isinstance(model, PreTrainedModel):
            kwargs['find_unused_parameters'] = not model.is_gradient_checkpointing
        else:
            kwargs['find_unused_parameters'] = True
        if self.args.ddp_bucket_cap_mb is not None:
            kwargs['bucket_cap_mb'] = self.args.ddp_bucket_cap_mb
        if self.args.ddp_broadcast_buffers is not None:
            kwargs['broadcast_buffers'] = self.args.ddp_broadcast_buffers
        self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)
    return model