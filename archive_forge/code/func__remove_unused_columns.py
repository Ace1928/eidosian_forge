import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
from ..core import (
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments
from transformers import pipeline
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
def _remove_unused_columns(self, dataset: 'Dataset'):
    if not self.config.remove_unused_columns:
        return dataset
    self._set_signature_columns_if_needed()
    signature_columns = self._signature_columns
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    columns = [k for k in signature_columns if k in dataset.column_names]
    if version.parse(datasets.__version__) < version.parse('1.4.0'):
        dataset.set_format(type=dataset.format['type'], columns=columns, format_kwargs=dataset.format['format_kwargs'])
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)