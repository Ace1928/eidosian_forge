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
def _extract_input_info(self, model: GraphModule) -> Dict[str, Dict]:
    """
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the input information for each observer returns it

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns (str) to a dict with keys:
            "input_activation_per_channel_max" : maps to the per_channel max values
            "input_activation_per_channel_min" : maps to the per_channel min values
            "input_activation_global_max" : maps to the global max recorded
            "input_activation_global_min" : maps to the global min recorded
        """
    input_info: Dict[str, Dict] = {}
    for fqn, module in model.named_modules():
        if self._is_supported(module):
            pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            input_info[fqn] = {self.ACTIVATION_PREFIX + self.PER_CHANNEL_MAX_KEY: pre_obs.max_val, self.ACTIVATION_PREFIX + self.PER_CHANNEL_MIN_KEY: pre_obs.min_val, self.ACTIVATION_PREFIX + self.GLOBAL_MAX_KEY: max(pre_obs.max_val), self.ACTIVATION_PREFIX + self.GLOBAL_MIN_KEY: min(pre_obs.min_val)}
    return input_info