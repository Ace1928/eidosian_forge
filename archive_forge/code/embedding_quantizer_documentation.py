from __future__ import annotations
import copy
from typing import List, Set
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
just handling global spec for now