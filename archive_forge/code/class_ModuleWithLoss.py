import functools
import math
import os
import shutil
import sys
import time
import types
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.integrations import hp_params
from transformers.utils import is_accelerate_available
from packaging import version
import huggingface_hub.utils as hf_hub_utils
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, RandomSampler
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.trainer_pt_utils import (
from transformers.trainer_utils import (
from transformers.training_args import ParallelMode
from transformers.utils import (
from ..utils import logging
from .training_args import ORTOptimizerNames, ORTTrainingArguments
from .utils import (
class ModuleWithLoss(nn.Module):

    def __init__(self, model, args, label_smoother):
        super().__init__()
        self._original_model = model
        self.args = args
        self.label_smoother = label_smoother

    def forward(self, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs):
        return self.compute_model_plus_loss_internal(self._original_model, inputs, return_outputs)

    @property
    def module(self):
        """The original `torch.nn.Module` that this module wraps.
        This property provides access to methods and properties on the original module."""
        return self._original_model.module

    @property
    def config(self):
        return self._original_model.config