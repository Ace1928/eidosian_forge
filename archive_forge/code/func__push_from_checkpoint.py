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
def _push_from_checkpoint(self, checkpoint_folder):
    if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
        return
    if not self.args.hub_always_push and self.push_in_progress is not None and (not self.push_in_progress.is_done()):
        return
    output_dir = self.args.output_dir
    modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
    if is_peft_available():
        modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
    for modeling_file in modeling_files:
        if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
            shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)
    torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    if self.args.save_strategy == IntervalStrategy.STEPS:
        commit_message = f'Training in progress, step {self.state.global_step}'
    else:
        commit_message = f'Training in progress, epoch {int(self.state.epoch)}'
    model_push_job = upload_folder(repo_id=self.hub_model_id, folder_path=output_dir, commit_message=commit_message, token=self.args.hub_token, run_as_future=True, ignore_patterns=['_*', f'{PREFIX_CHECKPOINT_DIR}-*'])
    push_jobs = [model_push_job]
    if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
        path_in_repo = 'last-checkpoint' if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
        checkpoint_push = upload_folder(repo_id=self.hub_model_id, folder_path=checkpoint_folder, path_in_repo=path_in_repo, commit_message=commit_message + ', checkpoint', token=self.args.hub_token, run_as_future=True)
        push_jobs.append(checkpoint_push)
    if self.push_in_progress is None or self.push_in_progress.is_done():
        self.push_in_progress = PushInProgress(push_jobs)
    else:
        self.push_in_progress.jobs.extend(push_jobs)