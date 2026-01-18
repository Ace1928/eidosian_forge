from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union, List
import torch
from .fake_quantize import (
from .observer import (
from .qconfig import (
def _get_symmetric_qnnpack_qat_qconfig_mapping() -> QConfigMapping:
    """
    Return a QConfigMapping that uses `torch.ao.quantization.default_symmetric_qnnpack_qat_qconfig`
    as the default QConfig.
    """
    default_qconfig = default_symmetric_qnnpack_qat_qconfig
    return _get_default_qconfig_mapping_with_default_qconfig(True, 'qnnpack', default_qconfig)