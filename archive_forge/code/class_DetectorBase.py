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
class DetectorBase(ABC):
    """ Base Detector Module
    Any detector class should derive from this class.

    Concrete detectors should follow the same general API, which includes:
    - A method to calculate and return observer insertion points
        - Should return both the fqns and the Observer class to insert
    - A method to return a report based on the detector
        - Should return a str-based report and dict info in Tuple[str,Dict] format
    """

    def __init__(self):
        super().__init__()
        self.detector_config_info = None

    @abstractmethod
    def determine_observer_insert_points(self, model) -> Dict:
        """
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict.
            This dict maps string keys to detector specific information
        """
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        """ Returns the name of the current detector """
        pass

    @abstractmethod
    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        """ Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        pass

    def _get_targeting_node(self, prepared_fx_model: GraphModule, target_fqn: str) -> torch.fx.node.Node:
        """
        Takes in a GraphModule and the target_fqn and finds the node whose target is this fqn.

        If it's not found, it means it is most likely inside a fused layer
            We just go one layer up in terms of the fqn we are searching for until we find parent node
            If we get to empty string, then we know that it doesn't exist

        The reason for the recursion is that if the model that we are looking for got fused,
        we will have module fqn as e.g. x.linear.0 but the graph will only have a node for the fused module,
        which would have fqn as x.linear so they will not match.
        To handle this, if we don't match, we then take off the last bit of the fqn e.g. x.linear.0 -> x.linear,
        or more generally foo.bar.baz -> foo.bar and search again, this will allow us to locate the correct module
        even in cases with fusion

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            target_fqn (str): The fqn of the layer we are trying to target

        Returns the node object we are trying to add observers around
        """
        for node in prepared_fx_model.graph.nodes:
            if node.target == target_fqn:
                return node
        parent_fqn_sep_index = target_fqn.rfind('.')
        if parent_fqn_sep_index == -1:
            raise ValueError("passed in target_fqn not found in graph's targets.")
        else:
            return self._get_targeting_node(prepared_fx_model, target_fqn[:parent_fqn_sep_index])

    @abstractmethod
    def generate_detector_report(self, model) -> Tuple[str, Dict[str, Any]]:
        """
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Tuple of two elements:
            Str: string report of the suggested improvements
            Dict: contains useful data collected by the observer pertinent to this report
        """
        pass