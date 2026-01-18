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
def _early_stop(self, policykl):
    """
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL

        Returns:
            `bool`: whether to early stop or not
        """
    early_stop = False
    if not self.config.early_stopping:
        return early_stop
    if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
        self.optimizer.zero_grad()
        early_stop = True
    elif self.is_distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.all_reduce(policykl, dist.ReduceOp.SUM)
        policykl /= self.accelerator.num_processes
        if policykl > 1.5 * self.config.target_kl:
            self.optimizer.zero_grad()
            early_stop = True
    return early_stop