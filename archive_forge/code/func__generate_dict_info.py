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
def _generate_dict_info(self, input_info: Dict, weight_info: Dict, comp_stats: Dict) -> Dict[str, Dict]:
    """
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            input_info (dict): A dict mapping each module to input range information
            weight_info (dict): A dict mapping each module to weight range information
            comp_stats (dict): A dict mapping each module to its corresponding comp stat

        Returns a dictionary mapping each module with relevant ModelReportObservers around them to:
            whether input weight equalization is recommended
            their s_c metric compared to the threshold
            the threshold used to make the recommendation
            the channel used for recording data
            the input channel range info
            the weight channel range info
        """
    input_weight_equalization_info: Dict[str, Dict] = {}
    for module_fqn in input_info:
        mod_input_info: Dict = input_info[module_fqn]
        mod_weight_info: Dict = weight_info[module_fqn]
        mod_comp_stat: Dict = comp_stats[module_fqn]
        channel_rec_vals: list = []
        for val in mod_comp_stat:
            float_rep: float = val.item()
            recommended: bool = float_rep >= self.ratio_threshold and float_rep <= 1 / self.ratio_threshold
            channel_rec_vals.append(recommended)
        input_weight_equalization_info[module_fqn] = {self.RECOMMENDED_KEY: channel_rec_vals, self.COMP_METRIC_KEY: mod_comp_stat, self.THRESHOLD_KEY: self.ratio_threshold, self.CHANNEL_KEY: self.ch_axis, **mod_input_info, **mod_weight_info}
    return input_weight_equalization_info