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
def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
    if self.config.kl_penalty == 'kl':
        return logprob - ref_logprob
    if self.config.kl_penalty == 'abs':
        return (logprob - ref_logprob).abs()
    if self.config.kl_penalty == 'mse':
        return 0.5 * (logprob - ref_logprob).square()
    if self.config.kl_penalty == 'full':
        return F.kl_div(ref_logprob, logprob, log_target=True, reduction='none').sum(-1)
    raise NotImplementedError