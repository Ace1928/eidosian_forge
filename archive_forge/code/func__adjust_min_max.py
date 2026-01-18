import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _adjust_min_max(self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    downsample_rate = int(torch.ceil((combined_max - combined_min) * upsample_rate / (self.max_val - self.min_val)).item())
    e = downsample_rate * (self.max_val - self.min_val) / upsample_rate - (combined_max - combined_min)
    start_idx = int(torch.round((self.min_val - combined_min) * self.bins * upsample_rate / (self.max_val - self.min_val)).item())
    combined_max = combined_max + e
    combined_min = combined_min
    return (combined_min, combined_max, downsample_rate, start_idx)