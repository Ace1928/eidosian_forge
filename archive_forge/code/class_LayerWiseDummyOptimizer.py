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
from .utils import (
class LayerWiseDummyOptimizer(torch.optim.Optimizer):
    """
    For Layer-wise optimizers such as GaLoRE optimizer, the optimization
    step is already done through the post gradient hooks. Therefore
    the trick is to create a dummy optimizer that can take arbitrary
    args and kwargs and return a no-op during training.

    Initial idea from @hiyouga in LLaMA-Factory:
    https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
    """

    def __init__(self, optimizer_dict=None, *args, **kwargs):
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {'lr': kwargs.get('lr', 0.001)})

    def zero_grad(self, set_to_none: bool=True) -> None:
        pass

    def step(self, closure=None) -> Optional[float]:
        pass