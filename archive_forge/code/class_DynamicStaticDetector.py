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
class DynamicStaticDetector(DetectorBase):
    """
    Determines whether dynamic or static quantization is more appropriate for a given module.

    Takes advantage of the ModelReportObserver that records range information.
    Stationary distribution of data are strictly above tolerance level for the comparison statistic:

        S = average_batch_activation_range/epoch_activation_range

    Nonstationary distributions are below or at the tolerance level for this metric.

    If the distribution of data right after the module is non-stationary, recommend dynamic quantization
        Otherwise recommend static quantization

    Args:
        tolerance (float, optional): The threshold where S metric is stationary above and non-stationary otherwise. Default: 0.5
    """
    DEFAULT_PRE_OBSERVER_NAME = 'model_report_pre_observer'
    DEFAULT_POST_OBSERVER_NAME = 'model_report_post_observer'
    STATIONARY_STR = 'stationary'
    NON_STATIONARY_STR = 'non-stationary'
    INPUT_ACTIVATION_PREFIX = 'input_activation_'
    OUTPUT_ACTIVATION_PREFIX = 'output_activation_'
    TOLERANCE_KEY = 'dynamic_static_tolerance'
    DEFAULT_DYNAMIC_REC_KEY = 'dynamic_recommended'
    PRE_OBS_COMP_STAT_KEY = INPUT_ACTIVATION_PREFIX + 'dynamic_static_comp_stat'
    POST_OBS_COMP_STAT_KEY = OUTPUT_ACTIVATION_PREFIX + 'dynamic_static_comp_stat'
    PRE_OBS_DATA_DIST_KEY = INPUT_ACTIVATION_PREFIX + 'dynamic_static_data_classification'
    POST_OBS_DATA_DIST_KEY = OUTPUT_ACTIVATION_PREFIX + 'dynamic_static_data_classification'
    IS_CURRENTLY_SUPPORTED_KEY = 'is_dynamic_supported'
    DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED = {nn.Linear}
    DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED = {nn.Conv1d, nn.Conv2d, nn.Conv3d}

    def __init__(self, tolerance=0.5):
        super().__init__()
        self.tolerance = tolerance
        self.useful_observer_fqns: Set[str] = set()

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        """
        Determines where observers need to be inserted for the Dynamic vs Static detector.
        For this detector, we want to place observers on either side of linear layers in the model.

        Currently inserts observers for:
            linear layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """
        obs_ctr = ModelReportObserver
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}
        for fqn, module in prepared_fx_model.named_modules():
            if self._is_supported(module, insert=True):
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)
                pre_obs_fqn = fqn + '.' + self.DEFAULT_PRE_OBSERVER_NAME
                obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
                post_obs_fqn = fqn + '.' + self.DEFAULT_POST_OBSERVER_NAME
                obs_fqn_to_info[post_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(), DETECTOR_IS_POST_OBS_KEY: True, DETECTOR_OBS_ARGS_KEY: (targeted_node,)}
        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        """ returns the string name of this detector"""
        return 'dynamic_vs_static_detector'

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        """ Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        dynamic_static_info = self._generate_dict_info(model)
        module_fqn_to_detector_qconfig_info = {}
        for module_fqn in dynamic_static_info:
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
            dynamic_static_recommended: bool = dynamic_static_info[module_fqn][self.DEFAULT_DYNAMIC_REC_KEY]
            detector_qconfig_info.is_activation_dynamic = dynamic_static_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
        return module_fqn_to_detector_qconfig_info

    def _is_supported(self, module: nn.Module, insert: bool=False) -> bool:
        """Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        is_supported_type = sum([isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED]) > 0
        future_supported_type = sum([isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED]) > 0
        supported = is_supported_type or future_supported_type
        if insert:
            return supported
        else:
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME) and hasattr(module, self.DEFAULT_POST_OBSERVER_NAME)
            return supported and has_obs

    def _generate_dict_info(self, model: GraphModule) -> Dict[str, Any]:
        """
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
                whether it is currently supported or planned for the future
        """
        module_dynamic_static_info = {}
        for fqn, module in model.named_modules():
            if self._is_supported(module):
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                post_obs = getattr(module, self.DEFAULT_POST_OBSERVER_NAME)
                pre_stat = pre_obs.get_batch_to_epoch_ratio()
                post_stat = post_obs.get_batch_to_epoch_ratio()
                dynamic_recommended = post_stat <= self.tolerance
                pre_obs_dist_classif = self.STATIONARY_STR if pre_stat > self.tolerance else self.NON_STATIONARY_STR
                post_obs_dist_classif = self.STATIONARY_STR if post_stat > self.tolerance else self.NON_STATIONARY_STR
                is_supported_type = sum([isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED]) > 0
                module_info = {self.TOLERANCE_KEY: self.tolerance, self.DEFAULT_DYNAMIC_REC_KEY: dynamic_recommended, self.PRE_OBS_COMP_STAT_KEY: pre_stat, self.PRE_OBS_DATA_DIST_KEY: pre_obs_dist_classif, self.POST_OBS_COMP_STAT_KEY: post_stat, self.POST_OBS_DATA_DIST_KEY: post_obs_dist_classif, self.IS_CURRENTLY_SUPPORTED_KEY: is_supported_type}
                module_dynamic_static_info[fqn] = module_info
        return module_dynamic_static_info

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        """
        Determines whether dynamic or static quantization is more appropriate for a given module.

        Takes advantage of the ModelReportObserver that records range information.
        Stationary distribution of data are strictly above tolerance level for the comparison statistic:

            S = average_batch_activation_range/epoch_activation_range

        Nonstationary distributions are below or at the tolerance level for this metric.

        If the distribution of data right after the module is non-stationary, recommend dynamic quantization
            Otherwise recommend static quantization

        This will then generate suggestions for dynamic vs static quantization focused around Linear.

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of of whether dynamic or static quantization is recommended for certain modules
            Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
                whether it is currently supported or planned for the future
        """
        module_dynamic_static_info = self._generate_dict_info(model)
        dynamic_vs_static_string = 'Dynamic vs. Static Quantization suggestions: \n'
        modules_added: bool = False
        dynamic_benefit = ' You will get more accurate results if you use dynamic quantization'
        static_benefit = ' You can increase model efficiency if you use static quantization'
        future_support_str = '. This layer is not yet supported for dynamic quantization'
        for module_fqn in module_dynamic_static_info.keys():
            modules_added = True
            module_info = module_dynamic_static_info[module_fqn]
            suggestion_string_template = 'For module {} it is suggested to use {} quantization because {}.\n'
            quantization_type = ''
            quantization_reasoning = 'the distribution of data before {} is {} and the distribution after is {}.'
            benefit_str = ''
            recommend_per_tensor = '. We recommend to add a {} before this module if it is static.'
            rec_lay_to_add = 'dynamic quantize per tensor layer'
            dynamic_per_tensor_string = recommend_per_tensor.format(rec_lay_to_add)
            dynamic_per_tensor_reasoning_string = ' This is because the input to this module has a non-stationary distribution'
            if module_info[self.DEFAULT_DYNAMIC_REC_KEY]:
                quantization_type = 'dynamic'
                benefit_str = dynamic_benefit
                if not module_info[self.IS_CURRENTLY_SUPPORTED_KEY]:
                    benefit_str += future_support_str
            else:
                quantization_type = 'static'
                benefit_str = static_benefit
            quantization_reasoning = quantization_reasoning.format(module_fqn, module_info[self.PRE_OBS_DATA_DIST_KEY], module_info[self.POST_OBS_DATA_DIST_KEY]) + benefit_str
            if module_info[self.PRE_OBS_DATA_DIST_KEY] == self.NON_STATIONARY_STR and module_info[self.POST_OBS_DATA_DIST_KEY] == self.STATIONARY_STR:
                quantization_reasoning = quantization_reasoning + dynamic_per_tensor_string + dynamic_per_tensor_reasoning_string
            module_suggestion_string = suggestion_string_template.format(module_fqn, quantization_type, quantization_reasoning)
            dynamic_vs_static_string += module_suggestion_string
        if not modules_added:
            dynamic_vs_static_string += 'No applicable layers for suggestions. Only linear and conv are valid.\n'
        return (dynamic_vs_static_string, module_dynamic_static_info)