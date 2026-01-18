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
        