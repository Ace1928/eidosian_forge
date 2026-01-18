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
def _load_optimizer_and_scheduler(self, checkpoint):
    """If optimizer and scheduler states exist, load them."""
    if checkpoint is None:
        return
    if self.is_deepspeed_enabled:
        if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            reissue_pt_warnings(caught_warnings)
        return
    checkpoint_file_exists = glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + '_*') if is_sagemaker_mp_enabled() else os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME)) or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN)) or (os.path.isdir(checkpoint) and any((OPTIMIZER_NAME_BIN.split('.')[0] in folder_name for folder_name in os.listdir(checkpoint) if os.path.isdir(os.path.join(checkpoint, folder_name)))))
    if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
        if is_torch_tpu_available():
            optimizer_state = torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location='cpu')
            with warnings.catch_warnings(record=True) as caught_warnings:
                lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location='cpu')
            reissue_pt_warnings(caught_warnings)
            xm.send_cpu_data_to_device(optimizer_state, self.args.device)
            xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)
            self.optimizer.load_state_dict(optimizer_state)
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        else:
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(checkpoint, 'user_content.pt')):

                    def opt_load_hook(mod, opt):
                        opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))
                else:

                    def opt_load_hook(mod, opt):
                        if IS_SAGEMAKER_MP_POST_1_10:
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True, back_compat=True))
                        else:
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))
                self.model_wrapped.register_post_step_hook(opt_load_hook)
            else:
                map_location = self.args.device if self.args.world_size > 1 else 'cpu'
                if self.is_fsdp_enabled:
                    load_fsdp_optimizer(self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, checkpoint, **_get_fsdp_ckpt_kwargs())
                else:
                    self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            reissue_pt_warnings(caught_warnings)