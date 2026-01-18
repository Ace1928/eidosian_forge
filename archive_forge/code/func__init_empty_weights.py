from __future__ import annotations
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from peft.config import PeftConfig
from peft.utils import (
from .tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge, check_target_module_exists
def _init_empty_weights(self, cls, *args, **kwargs) -> None:
    kwargs = kwargs.copy()
    final_device = kwargs.pop('device', 'cpu')
    cls.__init__(self, *args, device='meta', **kwargs)
    self.to_empty(device=final_device)