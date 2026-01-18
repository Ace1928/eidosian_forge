from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _generate_qconfig_mapping_helper(self, detector_qconfig_info_combined: Dict[str, DetectorQConfigInfo], generation_function: Callable) -> QConfigMapping:
    """
        This helper takes in the compiled detector qconfig info that
        has been compiled together and merges it into a QConfigMapping
        """
    qconfig_mapping = QConfigMapping()
    for fqn, module in self._model.named_modules():
        if fqn in detector_qconfig_info_combined:
            qconfig_info_compiled = detector_qconfig_info_combined[fqn]
            generated_qconfig = generation_function(qconfig_info_compiled, module)
            qconfig_mapping.set_module_name(fqn, generated_qconfig)
    return qconfig_mapping