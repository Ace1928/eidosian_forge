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
class PerChannelDetector(DetectorBase):
    """ This class is used to detect if any Linear or Conv layers in a model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        per_channel quantization can lead to major benefits in the form of accuracy.
        Therefore, if the backend used by the user supports it, it is recommended to use

        Args:
            backend (str, optional): the backend the user wishes to use in production
                Default value is current torch.backends.quantized.engine
    """
    BACKEND_KEY = 'backend'
    PER_CHAN_SUPPORTED_KEY = 'per_channel_quantization_supported'
    PER_CHAN_USED_KEY = 'per_channel_quantization_used'
    DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: Dict[str, Set[Any]] = {'fbgemm': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}, 'qnnpack': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}, 'onednn': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}, 'x86': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}}

    def __init__(self, backend: str=torch.backends.quantized.engine):
        super().__init__()
        self.backend_chosen = backend
        self.supported_modules = set()
        if self.backend_chosen in self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
            self.supported_modules = self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[self.backend_chosen]
        else:
            raise ValueError(f'Not configured to work with {self.backend_chosen}. Try a different default backend')

    def get_detector_name(self) -> str:
        """ returns the string name of this detector"""
        return 'per_channel_detector'

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        """ Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        per_channel_info = self._detect_per_channel_helper(model)
        module_fqn_to_detector_qconfig_info = {}
        for module_fqn in per_channel_info:
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
            per_chan_supported: bool = per_channel_info[module_fqn][self.PER_CHAN_SUPPORTED_KEY]
            detector_qconfig_info.is_weight_per_channel = per_chan_supported
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(self, model: nn.Module) -> Dict:
        """
        There is no observers inserted for the PerChannelDetector.

        Returns an empty dictionary since no observers are added or needed
        """
        return {}

    def _detect_per_channel_helper(self, model: nn.Module):
        """
        determines if per_channel quantization is supported in modules and submodules.

        Returns a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            model: The current module that is being checked to see if it is per_channel quantizable

        Returns dictionary mapping fqns to if per_channel quantization is possible
        """
        per_channel_info: Dict = {}
        for fqn, module in model.named_modules():
            is_in_include_list = sum([isinstance(module, x) for x in self.supported_modules]) > 0
            per_channel_supported = False
            if is_in_include_list:
                per_channel_supported = True
                q_config_file = module.qconfig
                assert isinstance(q_config_file, QConfig)
                q_or_s_obj = module.qconfig.weight.p.func()
                assert isinstance(q_or_s_obj, (FakeQuantize, ObserverBase))
                per_channel_used = False
                if hasattr(q_or_s_obj, 'ch_axis'):
                    if isinstance(q_or_s_obj, FakeQuantize):
                        if hasattr(q_or_s_obj, 'is_per_channel') and q_or_s_obj.is_per_channel:
                            per_channel_used = True
                    elif isinstance(q_or_s_obj, ObserverBase):
                        per_channel_used = True
                    else:
                        raise ValueError('Should be either observer or fake quant')
                per_channel_info[fqn] = {self.PER_CHAN_SUPPORTED_KEY: per_channel_supported, self.PER_CHAN_USED_KEY: per_channel_used, self.BACKEND_KEY: self.backend_chosen}
        return per_channel_info

    def generate_detector_report(self, model: nn.Module) -> Tuple[str, Dict[str, Any]]:
        """Checks if any Linear or Conv layers in the model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        Looks at q_config format and backend to determine if per_channel can be utilized.
        Uses the DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES structure to determine support

        Args:
            model: The prepared and calibrated model we want to check if using per_channel

        Returns a tuple with two elements:
            String report of potential actions to improve model (if per_channel quantization is available in backend)
            Dictionary mapping per_channel quantizable elements to:
                whether per_channel quantization is supported by the backend
                if it is being utilized in the current model
        """
        per_channel_info = self._detect_per_channel_helper(model)
        further_optims_str = f'Further Optimizations for backend {self.backend_chosen}: \n'
        optimizations_possible = False
        for fqn in per_channel_info:
            fqn_dict = per_channel_info[fqn]
            if fqn_dict[self.PER_CHAN_SUPPORTED_KEY] and (not fqn_dict[self.PER_CHAN_USED_KEY]):
                optimizations_possible = True
                further_optims_str += f'Module {fqn} can be configured to use per_channel quantization.\n'
        if optimizations_possible:
            further_optims_str += 'To use per_channel quantization, make sure the qconfig has a per_channel weight observer.'
        else:
            further_optims_str += 'No further per_channel optimizations possible.'
        return (further_optims_str, per_channel_info)