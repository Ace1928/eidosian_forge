from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.fx._equalize import (
from torch.ao.quantization.observer import _is_activation_post_process
def _supports_insertion(self, module: nn.Module) -> bool:
    """Returns whether the given module is supported for observers insertion

        Any module that doesn't have children and isn't an observer itself is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
    num_children = len(list(module.children()))
    return num_children == 0 and (not _is_activation_post_process(module))