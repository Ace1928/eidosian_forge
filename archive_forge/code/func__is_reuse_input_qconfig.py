from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _is_reuse_input_qconfig(qconfig: Optional[QConfig]):
    return qconfig is not None and isinstance(qconfig.activation(), ReuseInputObserver) and isinstance(qconfig.weight(), NoopObserver)