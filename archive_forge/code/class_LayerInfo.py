import logging
from typing import List, Tuple
import torch
import torch.nn as nn
class LayerInfo:
    """
    A class to record the layer attributes.
    """

    def __init__(self, name: str, layer: nn.Module, scale: float=1.0, scale_layer: bool=False) -> None:
        """
        layer_name: name of the layer e.g. fc1, conv1, relu1
        layer: type of the layer e.g. Linear, Conv2d, ReLU
        scaling_factor: user configurable scaling factor for the layer, defaults to 1.0
        found_inf_or_nan: a boolean indicating if any parameter of layer's gradient contains inf/nan
        growth_tracker: tracks number of step since last time scale was increased
        scale_layer: a boolean indicating if the layer should be scaled or not
        """
        self.layer_name = name
        self.layer = layer
        self.scaling_factor = scale
        self.found_inf_or_nan = False
        self.growth_tracker = 0
        self.scale_layer = scale_layer