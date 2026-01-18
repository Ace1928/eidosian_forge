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
def _supports_report_gen(self, module: nn.Module) -> bool:
    """Returns whether the given module is supported for report generation

        Any module that has a model report pre-observer is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
    return hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)