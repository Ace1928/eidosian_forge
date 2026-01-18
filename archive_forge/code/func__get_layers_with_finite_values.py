import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def _get_layers_with_finite_values(self) -> List[LayerInfo]:
    layers_with_finite_values: List = []
    for item in self.layer_info:
        if not item.found_inf_or_nan:
            layers_with_finite_values.append(item)
    return layers_with_finite_values