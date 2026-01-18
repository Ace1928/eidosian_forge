import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_tpu_available, is_training_run_on_sagemaker, logging
def _get_learning_rate(self):
    if self.is_deepspeed_enabled:
        try:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        except AssertionError as e:
            if 'need to call step' in str(e):
                logger.warning('tried to get lr value before scheduler/optimizer started stepping, returning lr=0')
                last_lr = 0
            else:
                raise
    else:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]['lr']
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
    return last_lr