from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _generate_module_fqn_to_detector_info_mapping(self, update_qconfig_info_function: Callable) -> Dict[str, DetectorQConfigInfo]:
    """
        Generates a QConfigMapping based on the suggestions of the
        ModelReport API. The generated mapping encompasses all the
        different types of feedback from the different detectors
        all into one place.

        These configs are based on the suggestions provided by the ModelReport API
        and can only be generated once the reports have been generated.

        Args:
            update_qconfig_info_function (Callable) takes in a function that takes in two DetectorQConfigInfo
            and updates the one that is being compiled

        Returns a Dict mapping module_fqns to DetectorQConfigInfo objects

        Note:
            Throws exception if we try to generate mapping on model we already removed observers from
            Throws exception if we try to generate mapping without preparing for callibration
        """
    if not self._prepared_flag:
        raise Exception('Cannot generate report without preparing model for callibration')
    if self._removed_observers:
        raise Exception('Cannot generate report on model you already removed observers from')
    detector_qconfig_info_combined: Dict[str, DetectorQConfigInfo] = {}
    for detector in self._desired_report_detectors:
        detector_info: Dict[str, DetectorQConfigInfo] = detector.get_qconfig_info(self._model)
        for module_fqn in detector_info:
            if module_fqn in detector_qconfig_info_combined:
                current_options = detector_qconfig_info_combined[module_fqn]
                detector_options = detector_info[module_fqn]
                update_qconfig_info_function(current_options, detector_options)
            else:
                detector_qconfig_info_combined[module_fqn] = detector_info[module_fqn]
    return detector_qconfig_info_combined