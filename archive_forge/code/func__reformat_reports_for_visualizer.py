from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _reformat_reports_for_visualizer(self) -> OrderedDict:
    """
        Takes the generated reports and reformats them into the format that is desired by the
        ModelReportVisualizer

        Returns an OrderedDict mapping module_fqns to their features
        """
    module_fqns_to_features: Dict[str, Dict] = {}
    for report_name in self._generated_reports:
        module_info = self._generated_reports[report_name]
        for module_fqn in module_info:
            if module_fqn in module_fqns_to_features:
                new_info: Dict = module_info[module_fqn]
                present_info: Dict = module_fqns_to_features[module_fqn]
                if self._is_same_info_for_same_key(new_info, present_info):
                    module_fqns_to_features[module_fqn] = {**new_info, **present_info}
                else:
                    error_str = 'You have the same key with different values across detectors. '
                    error_str += 'Someone incorrectly implemented a detector with conflicting keys to existing detectors.'
                    raise ValueError(error_str)
            else:
                module_fqns_to_features[module_fqn] = module_info[module_fqn]
    features_by_module: OrderedDict[str, Dict] = OrderedDict()
    for fqn, module in self._model.named_modules():
        if fqn in module_fqns_to_features:
            features_by_module[fqn] = module_fqns_to_features[fqn]
    return features_by_module