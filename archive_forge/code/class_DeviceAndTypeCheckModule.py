import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import random
from statistics import mean
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed
class DeviceAndTypeCheckModule(Base):
    """A simple module for checking Tensor devices and dtypes."""

    def __init__(self, expected_input_dtype: Optional[torch.dtype]=None, expected_input_device: Optional[torch.device]=None, expected_param_dtype: Optional[torch.dtype]=None, expected_param_device: Optional[torch.device]=None, expected_loss_dtype: Optional[torch.dtype]=None, expected_loss_device: Optional[torch.device]=None, expected_buffer_dtype: Optional[torch.device]=None):
        super().__init__()
        self.expected_input_dtype = expected_input_dtype
        self.expected_input_device = expected_input_device
        self.expected_param_dtype = expected_param_dtype
        self.expected_param_device = expected_param_device
        self.expected_loss_dtype = expected_loss_dtype
        self.expected_loss_device = expected_loss_device
        self.expected_buffer_dtype = expected_buffer_dtype
        self.linear = nn.Linear(5, 5)
        self.register_buffer('buffer', torch.rand((5,)))

    def _check(self, key: str, x: Union[torch.device, torch.dtype], expected: Union[Optional[torch.device], Optional[torch.dtype]]) -> None:
        assert expected in {None, x}, f'{key} ({x}) != expected ({expected})'

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        x = input[0]
        self._check('input.dtype', x.dtype, self.expected_input_dtype)
        self._check('input.device', x.device, self.expected_input_device)
        param = self.linear.weight
        self._check('param.dtype', param.dtype, self.expected_param_dtype)
        self._check('param.device', param.device, self.expected_param_device)
        self._check('buffer.dtype', self.buffer.dtype, self.expected_buffer_dtype)
        x = x + self.buffer
        loss = (self.linear(x) + self.buffer).sum()
        self._check('loss.dtype', loss.dtype, self.expected_loss_dtype)
        self._check('loss.device', loss.device, self.expected_loss_device)
        return loss