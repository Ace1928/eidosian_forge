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
def _add_sm_patterns_to_gitignore(self) -> None:
    """Add SageMaker Checkpointing patterns to .gitignore file."""
    if not self.is_world_process_zero():
        return
    patterns = ['*.sagemaker-uploading', '*.sagemaker-uploaded']
    if os.path.exists(os.path.join(self.repo.local_dir, '.gitignore')):
        with open(os.path.join(self.repo.local_dir, '.gitignore'), 'r') as f:
            current_content = f.read()
    else:
        current_content = ''
    content = current_content
    for pattern in patterns:
        if pattern not in content:
            if content.endswith('\n'):
                content += pattern
            else:
                content += f'\n{pattern}'
    if content != current_content:
        with open(os.path.join(self.repo.local_dir, '.gitignore'), 'w') as f:
            logger.debug(f'Writing .gitignore file. Content: {content}')
            f.write(content)
    self.repo.git_add('.gitignore')
    time.sleep(0.5)
    if not self.repo.is_repo_clean():
        self.repo.git_commit('Add *.sagemaker patterns to .gitignore.')
        self.repo.git_push()