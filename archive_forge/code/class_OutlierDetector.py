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
class OutlierDetector(DetectorBase):
    """
    Determines whether there are significant outliers in activation data around a certain layer.

    This is ideally used in conjunction with information on stationary vs. non-stationary distribution:
        If the data is stationary, and there are significant outliers, then we want to flag them
        We want to do this on a per channel basis for detecting outliers

    Determines whether activation data is flagged as outlier based on if data is stationary and:
        p_r = avg(100th percentile / "reference_percentile"th percentile)
        where:
            p_r is average percentile ratio across all batches in the epoch
            reference_percentile is a percentile values between 0 and 100 exclusive

        if p_r is above some threshold, then we consider the activations to have significant outliers

    Args:
        ratio_threshold (float, optional): The threshold for p_r to determine if there are outliers in activations
            Should be >= 1
            Default: 3.5
        reference_percentile (float, optional): The denominator to find the relative scale of the 100th percentile
            Should be between 0 and 1
            Default: 0.975
        fraction_batches_used_threshold (float, optional): Threshold of fraction of batches per channel to determine outlier
            If fraction is below this, we deem number of samples used to calculate outliers as insignificant and alert user
            regardless of whether we detected outliers or not in channel to take a closer look at channel results
            Should be between 0 and 1
            Default: 0.95
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for p_r to determine if there are outliers in activations
        The p_r value (average ratio of 100th percentile/reference_percentile) is compared to ratio_threshold
        If it is significantly greater, then we consider it an outlier
        This threshold was calculated based on the ratio of the percentiles in a normal distribution
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`reference_percentile`: The denominator of the top fraction to find the relative scale of the 100th percentile
        Should be between 0 and 1
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`fraction_batches_used_threshold`: The fraction of batches to determine outliers for each channel should be above this
        Some batches may not be used because of 0-based errors, so this is to ensure a good amount of the total batches are used
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine outliers

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """
    DEFAULT_PRE_OBSERVER_NAME: str = 'model_report_pre_observer'
    INPUT_ACTIVATION_PREFIX = 'input_activation_'
    OUTLIER_KEY = 'outliers_detected'
    NUM_BATCHES_KEY = 'outlier_detection_batches_used'
    IS_SUFFICIENT_BATCHES_KEY = 'outlier_detection_is_sufficient_batches'
    COMP_METRIC_KEY = 'outlier_detection_percentile_ratios'
    RATIO_THRES_KEY = 'outlier_detection_ratio_threshold'
    REF_PERCENTILE_KEY = 'outlier_detection_reference_percentile'
    CHANNEL_AXIS_KEY = 'outlier_detection_channel_axis'
    MAX_VALS_KEY = INPUT_ACTIVATION_PREFIX + 'per_channel_max'
    CONSTANT_COUNTS_KEY = 'constant_batch_counts'

    def __init__(self, ratio_threshold: float=3.5, reference_percentile: float=0.975, fraction_batches_used_threshold: float=0.95, ch_axis: int=1):
        self.ratio_threshold = ratio_threshold
        assert reference_percentile >= 0 and reference_percentile <= 1
        assert fraction_batches_used_threshold >= 0 and fraction_batches_used_threshold <= 1
        self.reference_percentile = reference_percentile
        self.fraction_batches_used_threshold = fraction_batches_used_threshold
        self.ch_axis = ch_axis

    def get_detector_name(self) -> str:
        """Returns the name of this detector"""
        return 'outlier_detector'

    def _supports_insertion(self, module: nn.Module) -> bool:
        """Returns whether the given module is supported for observers insertion

        Any module that doesn't have children and isn't an observer itself is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
        num_children = len(list(module.children()))
        return num_children == 0 and (not _is_activation_post_process(module))

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        """ Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        return {}

    def _supports_report_gen(self, module: nn.Module) -> bool:
        """Returns whether the given module is supported for report generation

        Any module that has a model report pre-observer is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
        return hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        """ Determines where observers need to be inserted for the Outlier Detector.

        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            all layers that do not have children (leaf level layers)

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
            if self._supports_insertion(module):
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)
                pre_obs_fqn = fqn + '.' + self.DEFAULT_PRE_OBSERVER_NAME
                obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis, comp_percentile=self.reference_percentile), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
        return obs_fqn_to_info

    def _calculate_outlier_info(self, percentile_ratios: torch.Tensor, counted_batches: torch.Tensor, total_batches: int) -> Dict[str, List[bool]]:
        """
        Gives info on whether the percentile ratios calculated would be considered outliers
        Also gives information on whether the collected data is statistically significant to make this claim

        Args:
            percentile_ratios (torch.Tensor): The average percentile_ratios per channel calculated by the observer
            counted_batches (torch.Tensor): The number of batches used for average calculation per tensor
            total_batches (int): The total number of batches that passed through observer in this epoch

        Returns a dictionary mapping:
            "outliers_detected" : list of bools per channel that are true if it is considered an outlier
            "is_sufficient_batches": if o_r was >= fraction_batches_used_threshold:
                where o_r = counted_batches / total_batches
        """
        outlier_dict: Dict[str, List[bool]] = {self.OUTLIER_KEY: [], self.IS_SUFFICIENT_BATCHES_KEY: []}
        ratios_list: List = percentile_ratios.tolist()
        num_batches_list: List = counted_batches.tolist()
        significant_size = [batch_size / total_batches >= self.fraction_batches_used_threshold for batch_size in num_batches_list]
        outlier_dict[self.IS_SUFFICIENT_BATCHES_KEY] = significant_size
        outlier_detected = [ratio > self.ratio_threshold for ratio in ratios_list]
        outlier_dict[self.OUTLIER_KEY] = outlier_detected
        return outlier_dict

    def _generate_info_dict(self, model: GraphModule) -> Dict[str, Dict]:
        """
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns to:
            whether there were outliers found in activation before
            the number of batches used for each channel
            whether fraction of applicable batches used is above fraction_batches_used_threshold
            their p_r metric compared to the threshold
            the threshold used to make the recommendation
            the reference_percentile used to make the recommendation
            the channel axis used to determine individual channels
            the constant batch counts per channel
            the per channel max values
        """
        info_dict: Dict[str, Dict] = {}
        for fqn, module in model.named_modules():
            if self._supports_report_gen(module):
                pre_obs: ModelReportObserver = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                num_batches: torch.Tensor = pre_obs.percentile_batches_tracked
                average_ratios: torch.Tensor = pre_obs.average_percentile_ratio
                channel_batch_cnts: torch.Tensor = pre_obs.constant_channels
                total_batches: int = pre_obs.num_batches_tracked
                max_vals: torch.Tensor = pre_obs.max_val
                for index, ratio_val in enumerate(average_ratios):
                    if ratio_val.item() < 0:
                        average_ratios[index] = -ratio_val
                    if ratio_val.item() < 1:
                        average_ratios[index] = 1 / ratio_val
                outlier_calcs = self._calculate_outlier_info(average_ratios, num_batches, total_batches)
                info_dict[fqn] = {self.CHANNEL_AXIS_KEY: self.ch_axis, self.REF_PERCENTILE_KEY: self.reference_percentile, self.RATIO_THRES_KEY: self.ratio_threshold, self.COMP_METRIC_KEY: average_ratios, self.NUM_BATCHES_KEY: num_batches, self.OUTLIER_KEY: outlier_calcs[self.OUTLIER_KEY], self.IS_SUFFICIENT_BATCHES_KEY: outlier_calcs[self.IS_SUFFICIENT_BATCHES_KEY], self.CONSTANT_COUNTS_KEY: channel_batch_cnts, self.MAX_VALS_KEY: max_vals}
        return info_dict

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        """
        Determines whether input weight equalization is appropriate for a given module.

        Takes advantage of the ModelReport Observer which records the relevant percentile information

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of of whether there are outliers in the activations around certain modules
            Dictionary mapping modules of interest to:
                whether there were outliers found in activation before
                the number of batches used for each channel
                whether fraction of applicable batches used is above fraction_batches_used_threshold
                their p_r metric compared to the threshold
                the threshold used to make the recommendation
                the reference_percentile used to make the recommendation
                the channel axis used to determine individual channels
                the constant batch counts per channel
                the per channel max values
        """
        info_dict = self._generate_info_dict(model)
        outlier_string = 'Outlier detection report: \n'
        added_module: bool = False
        module_suggestion_str = 'For Module {} looked at with axis {}: \n'
        channel_suggestion_str = '\tFor channel {}, we found outliers in the preceding activation data with {}.\n'
        channel_max_value_str = 'a max value across all batches of {}'
        note_string = 'Note: outlier detection is only reliable for {}. We recommend {} to ensure the most accurate results.'
        note_distribution = 'stationary distributions'
        note_rec = 'running the static vs. dynamic detector to ensure activation data before modules above is stationary'
        constant_str = '\tFor channel {}, we found {} constant value batches. {}\n'
        constant_suggestion = 'We recommend taking a look at the dict and data to see how frequent this occurred and why.'
        for module_fqn in info_dict:
            mod_info: Dict[str, Any] = info_dict[module_fqn]
            added_model_desc = False
            for index, outlier_detected in enumerate(mod_info[self.OUTLIER_KEY]):
                if outlier_detected:
                    if not added_model_desc:
                        outlier_string += module_suggestion_str.format(module_fqn, self.ch_axis)
                        added_model_desc = True
                    added_module = True
                    max_value_found_str = channel_max_value_str.format(mod_info[self.MAX_VALS_KEY][index])
                    channel_str = channel_suggestion_str.format(index, max_value_found_str)
                    outlier_string += channel_str
                if mod_info[self.CONSTANT_COUNTS_KEY][index] != 0:
                    if not added_model_desc:
                        outlier_string += module_suggestion_str.format(module_fqn, self.ch_axis)
                        added_model_desc = True
                    constant_values_for_channel = mod_info[self.CONSTANT_COUNTS_KEY][index]
                    formatted_str = constant_str.format(index, constant_values_for_channel, constant_suggestion)
                    outlier_string += formatted_str
                    added_module = True
        if added_module:
            note_composed = note_string.format(note_distribution, note_rec)
            outlier_string += note_composed
        else:
            outlier_string += 'There were no outliers found in the activations.\n'
        return (outlier_string, info_dict)