from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union, List
import torch
from .fake_quantize import (
from .observer import (
from .qconfig import (
def _get_default_qconfig_mapping_with_default_qconfig(is_qat: bool, backend: str, default_qconfig: QConfig) -> QConfigMapping:
    """
    Return a QConfigMapping that uses the provided qconfig as the default QConfig.
    """
    if is_qat:
        qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    else:
        qconfig_mapping = get_default_qconfig_mapping(backend)
    qconfig_mapping.set_global(default_qconfig)
    for pattern in qconfig_mapping.object_type_qconfigs.keys():
        if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
            qconfig_mapping.set_object_type(pattern, default_qconfig)
    return qconfig_mapping