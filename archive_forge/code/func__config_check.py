import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from ..models import DDPOStableDiffusionPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
def _config_check(self) -> Tuple[bool, str]:
    samples_per_epoch = self.config.sample_batch_size * self.accelerator.num_processes * self.config.sample_num_batches_per_epoch
    total_train_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.train_gradient_accumulation_steps
    if not self.config.sample_batch_size >= self.config.train_batch_size:
        return (False, f'Sample batch size ({self.config.sample_batch_size}) must be greater than or equal to the train batch size ({self.config.train_batch_size})')
    if not self.config.sample_batch_size % self.config.train_batch_size == 0:
        return (False, f'Sample batch size ({self.config.sample_batch_size}) must be divisible by the train batch size ({self.config.train_batch_size})')
    if not samples_per_epoch % total_train_batch_size == 0:
        return (False, f'Number of samples per epoch ({samples_per_epoch}) must be divisible by the total train batch size ({total_train_batch_size})')
    return (True, '')