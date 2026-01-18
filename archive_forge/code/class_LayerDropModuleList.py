import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]]=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or dropout_probs[i] > self.p:
                yield m