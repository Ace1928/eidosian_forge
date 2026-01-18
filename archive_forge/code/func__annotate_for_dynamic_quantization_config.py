from __future__ import annotations
import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
def _annotate_for_dynamic_quantization_config(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    module_name_list = list(self.module_name_config.keys())
    for module_name, config in self.module_name_config.items():
        self._annotate_all_dynamic_patterns(model, config, _get_module_name_filter(module_name))
    tp_list = list(self.module_type_config.keys())
    for module_type, config in self.module_type_config.items():
        self._annotate_all_dynamic_patterns(model, config, _get_module_type_filter(module_type))
    self._annotate_all_dynamic_patterns(model, self.global_config, _get_not_module_type_or_name_filter(tp_list, module_name_list))
    return model