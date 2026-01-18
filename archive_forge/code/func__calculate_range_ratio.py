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
def _calculate_range_ratio(self, info_dict: Dict, info_str: str, module_fqn: str) -> torch.Tensor:
    """
        Takes in an info dict and calculates the s_c matrix.

        Args:
            info_dict (dict): A dictionary of either input or weight range info
            info_str (str): A str describing whether currently looking at weight or input info
                Either "weight" or "input"
            module_fqn (str): The fqn of the module we are looking at

        Returns a tensor of values, where each value is the s_c stat for a different channel
        """
    prefix_str = self.ACTIVATION_PREFIX if info_str == self.INPUT_STR else self.WEIGHT_PREFIX
    per_channel_range = info_dict[prefix_str + self.PER_CHANNEL_MAX_KEY] - info_dict[prefix_str + self.PER_CHANNEL_MIN_KEY]
    global_range = info_dict[prefix_str + self.GLOBAL_MAX_KEY] - info_dict[prefix_str + self.GLOBAL_MIN_KEY]
    if global_range == 0:
        range_zero_explanation = "We recommend removing this channel as it doesn't provide any useful information."
        raise ValueError('The range of the {} data for module {} is 0, which means you have a constant value channel. {}'.format(info_str, module_fqn, range_zero_explanation))
    ratio = per_channel_range / global_range
    return ratio