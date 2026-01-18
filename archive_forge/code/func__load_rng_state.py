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
def _load_rng_state(self, checkpoint):
    if checkpoint is None:
        return
    if self.args.world_size > 1:
        process_index = self.args.process_index
        rng_file = os.path.join(checkpoint, f'rng_state_{process_index}.pth')
        if not os.path.isfile(rng_file):
            logger.info(f"Didn't find an RNG file for process {process_index}, if you are resuming a training that wasn't launched in a distributed fashion, reproducibility is not guaranteed.")
            return
    else:
        rng_file = os.path.join(checkpoint, 'rng_state.pth')
        if not os.path.isfile(rng_file):
            logger.info("Didn't find an RNG file, if you are resuming a training that was launched in a distributed fashion, reproducibility is not guaranteed.")
            return
    checkpoint_rng_state = torch.load(rng_file)
    random.setstate(checkpoint_rng_state['python'])
    np.random.set_state(checkpoint_rng_state['numpy'])
    torch.random.set_rng_state(checkpoint_rng_state['cpu'])
    if torch.cuda.is_available():
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            torch.cuda.random.set_rng_state_all(checkpoint_rng_state['cuda'])
        else:
            try:
                torch.cuda.random.set_rng_state(checkpoint_rng_state['cuda'])
            except Exception as e:
                logger.info(f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}\nThis won't yield the same results as if the training had not been interrupted.")
    if is_torch_tpu_available():
        xm.set_rng_state(checkpoint_rng_state['xla'])
    if is_torch_npu_available():
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            torch.npu.random.set_rng_state_all(checkpoint_rng_state['npu'])
        else:
            try:
                torch.npu.random.set_rng_state(checkpoint_rng_state['npu'])
            except Exception as e:
                logger.info(f"Didn't manage to set back the RNG states of the NPU because of the following error:\n {e}\nThis won't yield the same results as if the training had not been interrupted.")