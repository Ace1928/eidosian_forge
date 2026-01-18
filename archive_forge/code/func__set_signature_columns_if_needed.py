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
def _set_signature_columns_if_needed(self):
    if self._signature_columns is None:
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        self._signature_columns += ['label', 'query', 'response']