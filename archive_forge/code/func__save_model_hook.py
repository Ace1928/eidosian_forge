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
def _save_model_hook(self, models, weights, output_dir):
    self.sd_pipeline.save_checkpoint(models, weights, output_dir)
    weights.pop()