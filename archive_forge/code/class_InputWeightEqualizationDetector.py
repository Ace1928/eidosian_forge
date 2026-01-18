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
class InputWeightEqualizationDetector(DetectorBase):
    """
    Determines whether input-weight equalization can help improve quantization for certain modules.

    Specifically, this list of modules includes:
        linear
        conv

    Determines whether input-weight equalization is recommended based on the comp stat:
        s_c = sqrt(w_c/W)/sqrt(i_c/I)
        where:
            w_c is range of weight for channel c, W is range of weight over all channels
            i_c is range of input for channel c, I is range of input over all channels

        if s_c >= threshold or <= 1 / threshold, recommends input-weight equalization

    Args:
        ratio_threshold (float): The threshold for s_c to determine if input-weight equalization is suggested
            Should be between 0 and 1 (both non-inclusive)
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for s_c to determine if input-weight equalization is suggested
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine input weight equalization

    * :attr:`SUPPORTED_MODULES`: This specifies the modules that are supported for input-weight equalization

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """
    SUPPORTED_MODULES: Set[Callable] = {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}
    DEFAULT_PRE_OBSERVER_NAME: str = 'model_report_pre_observer'
    WEIGHT_PREFIX = 'weight_'
    ACTIVATION_PREFIX = 'input_activation_'
    PER_CHANNEL_MAX_KEY = 'per_channel_max'
    PER_CHANNEL_MIN_KEY = 'per_channel_min'
    GLOBAL_MAX_KEY = 'global_max'
    GLOBAL_MIN_KEY = 'global_min'
    RECOMMENDED_KEY = 'input_weight_equalization_recommended'
    COMP_METRIC_KEY = 'input_weight_channel_comparison_metrics'
    THRESHOLD_KEY = 'input_weight_threshold'
    CHANNEL_KEY = 'input_weight_channel_axis'
    WEIGHT_STR = 'weight'
    INPUT_STR = 'input'
    DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO = 0.4

    def __init__(self, ratio_threshold: float, ch_axis: int=1):
        if ratio_threshold <= 0 or ratio_threshold >= 1:
            raise ValueError('Make sure threshold is > 0 and < 1')
        self.ratio_threshold: float = ratio_threshold
        self.ch_axis: int = ch_axis

    def _is_supported(self, module: nn.Module, insert: bool=False) -> bool:
        """Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        is_supported_type = sum([type(module) is x for x in self.SUPPORTED_MODULES]) > 0
        if insert:
            return is_supported_type
        else:
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            return is_supported_type and has_obs

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        """ Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        input_values: Dict[str, Dict] = self._extract_input_info(model)
        weight_values: Dict[str, Dict] = self._extract_weight_info(model)
        comp_stats: Dict[str, torch.Tensor] = self._generate_comparison_values(input_values, weight_values)
        input_weight_equalization_info: Dict[str, Dict] = self._generate_dict_info(input_values, weight_values, comp_stats)
        module_fqn_to_detector_qconfig_info = {}
        for module_fqn in input_weight_equalization_info:
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
            input_weight_recommended: bool = input_weight_equalization_info[module_fqn][self.RECOMMENDED_KEY]
            detector_qconfig_info.is_equalization_recommended = input_weight_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        """Determines where observers need to be inserted for the Input Weight Equalization Detector.
        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            linear layers
            conv layers

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
                obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        """Returns the name of this detector"""
        return 'input_weight_equalization_detector'

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

    def _extract_weight_info(self, model: GraphModule) -> Dict[str, Dict]:
        """
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the weight information for each layer an observer is attached to.

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """
        weight_info: Dict[str, Dict] = {}
        for fqn, module in model.named_modules():
            if self._is_supported(module):
                device = module.weight.device
                min_val: torch.Tensor = torch.tensor([float('inf')], device=device)
                max_val: torch.Tensor = torch.tensor([float('-inf')], device=device)
                x_copy = module.weight
                x_dim = x_copy.size()
                new_axis_list = [i for i in range(len(x_dim))]
                new_axis_list[self.ch_axis] = 0
                new_axis_list[0] = self.ch_axis
                y = x_copy.permute(new_axis_list)
                y = y.to(min_val.dtype)
                y = torch.flatten(y, start_dim=1)
                if min_val.numel() == 0 or max_val.numel() == 0:
                    min_val, max_val = torch.aminmax(y, dim=1)
                else:
                    min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                    min_val = torch.min(min_val_cur, min_val)
                    max_val = torch.max(max_val_cur, max_val)
                weight_info[fqn] = {self.WEIGHT_PREFIX + self.PER_CHANNEL_MAX_KEY: max_val, self.WEIGHT_PREFIX + self.PER_CHANNEL_MIN_KEY: min_val, self.WEIGHT_PREFIX + self.GLOBAL_MAX_KEY: max(max_val), self.WEIGHT_PREFIX + self.GLOBAL_MIN_KEY: min(min_val)}
        return weight_info

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

    def _generate_comparison_values(self, input_info: Dict, weight_info: Dict) -> Dict[str, torch.Tensor]:
        """
        Takes in the information on the min and max values of the inputs and weights and:
            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)

        Args:
            input_info (dict): A dict mapping each observer to input range information
            weight_info (dict): A dict mapping each observer to weight range information

        Returns a dict mapping relevant observer fqns (str) to a 1-D tensor.
            Each value is a different s_c value for a different channel
        """
        module_fqn_to_channel: Dict[str, torch.Tensor] = {}
        for module_fqn in input_info:
            if module_fqn not in weight_info:
                raise KeyError(f'Unable to find weight range stats for module {module_fqn}')
            weight_ratio = self._calculate_range_ratio(weight_info[module_fqn], self.WEIGHT_STR, module_fqn)
            input_ratio = self._calculate_range_ratio(input_info[module_fqn], self.INPUT_STR, module_fqn)
            weight_channels = len(weight_ratio)
            input_channels = len(input_ratio)
            if weight_channels != input_channels:
                assert input_channels % weight_channels == 0, 'input channels should be divisible by weight channels.'
                rep_factor: int = input_channels // weight_channels
                weight_ratio = weight_ratio.repeat(rep_factor)
            s = torch.sqrt(weight_ratio) / torch.sqrt(input_ratio)
            module_fqn_to_channel[module_fqn] = s
        return module_fqn_to_channel

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

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        """
        Determines whether input weight equalization is appropriate for a given module.

        Takes advantage of the ModelReport Observer which records per channel information of input range
        It then uses the passed in weight info inconjunction to compute the desired ratio
        Finally, it gives suggestions based on this information for each module of interest

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of of whether input weight equalization is recommended for certain modules
            Dictionary mapping modules of interest to:
                whether input weight equalization is recommended
                their s_c metric compared to the threshold
                the threshold used to make the recommendation
                the channel used for recording data
                the input channel range info
                the weight channel range info
        """
        input_values: Dict[str, Dict] = self._extract_input_info(model)
        weight_values: Dict[str, Dict] = self._extract_weight_info(model)
        comp_stats: Dict[str, torch.Tensor] = self._generate_comparison_values(input_values, weight_values)
        input_weight_equalization_info: Dict[str, Dict] = self._generate_dict_info(input_values, weight_values, comp_stats)
        input_weight_string = 'Input-Weight Equalization suggestions: \n'
        module_suggestion_str = 'For Module {} looked at with axis {}: \n'
        channel_suggestion_str = '\tWe suggest {} input weight equalization because {}\n'
        use_str = 'to use'
        no_use_str = 'to not use'
        input_weight_benefit_str = '{}/{} channels would benefit and we expect significant reduction in quantization error.'
        input_weight_non_benefit_reasoning = '{}/{} channels benefitting from input-weight equalization being applied.'
        input_weight_non_benefit_str = "we don't expect much improvement from input-weight equalization based on {}"
        added_module: bool = False
        for module_fqn in input_weight_equalization_info:
            added_module = True
            input_weight_string += module_suggestion_str.format(module_fqn, self.ch_axis)
            mod_info: Dict[str, Any] = input_weight_equalization_info[module_fqn]
            recommendation_per_channel: torch.Tensor = mod_info[self.RECOMMENDED_KEY]
            num_recs = sum(recommendation_per_channel)
            if num_recs / len(recommendation_per_channel) >= self.DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO:
                input_benefit_formatted = input_weight_benefit_str.format(num_recs, len(recommendation_per_channel))
                channel_str = channel_suggestion_str.format(use_str, input_benefit_formatted)
                input_weight_string += channel_str
            else:
                non_benefit_reason_formatted = input_weight_non_benefit_reasoning.format(num_recs, len(recommendation_per_channel))
                non_benefit_str = input_weight_non_benefit_str.format(non_benefit_reason_formatted)
                channel_str = channel_suggestion_str.format(no_use_str, non_benefit_str)
                input_weight_string += channel_str
        if not added_module:
            input_weight_string += 'No applicable layers for suggestions. Only linear and conv valid.\n'
        return (input_weight_string, input_weight_equalization_info)