import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import (
def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
    deepspeed_plugin = self.accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    if model is not None:
        if hasattr(model, 'config'):
            hidden_size = max(model.config.hidden_sizes) if getattr(model.config, 'hidden_sizes', None) else getattr(model.config, 'hidden_size', None)
            if hidden_size is not None and config_kwargs['zero_optimization']['stage'] == 3:
                config_kwargs.update({'zero_optimization.reduce_bucket_size': hidden_size * hidden_size, 'zero_optimization.stage3_param_persistence_threshold': 10 * hidden_size, 'zero_optimization.stage3_prefetch_bucket_size': 0.9 * hidden_size * hidden_size})
    if config_kwargs['zero_optimization']['stage'] != 3:
        config_kwargs['zero_optimization']['stage'] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model