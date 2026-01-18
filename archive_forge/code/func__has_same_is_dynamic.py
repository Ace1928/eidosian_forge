import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _has_same_is_dynamic(qspec_a: QuantizationSpecBase, qspec_b: QuantizationSpecBase):
    return hasattr(qspec_a, 'is_dynamic') and hasattr(qspec_b, 'is_dynamic') and (qspec_a.is_dynamic == qspec_b.is_dynamic)