import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def _update_scale(self) -> None:
    """
        For each layer, if an inf/nan is found, then multiply the scaling factor
        of that layer by the backoff factor and set the growth tracker of that
        layer to 0. Else, increment the growth tracker of the layer. If growth
        tracker equals the growth interval, then multiply the scaling factor of
        the layer by the growth factor and reset the layer's growth tracker to 0.
        Finally, clip the scaling factor to the range
        [self.min_scaling_factor, self.max_scaling_factor]. The min/max scaling
        factor values are user configurable.
        """
    if not self._apply_layerwise_scaling:
        return
    for layer in self.layer_info:
        if layer.found_inf_or_nan:
            if layer.scale_layer:
                layer.scaling_factor = max(self._min_scale, min(self._backoff_factor * layer.scaling_factor, self._max_scale))
                layer.growth_tracker = 0
        else:
            layer.growth_tracker += 1
            if layer.scale_layer and layer.growth_tracker == self._growth_interval:
                layer.scaling_factor = max(self._min_scale, min(self._growth_factor * layer.scaling_factor, self._max_scale))
                layer.growth_tracker = 0