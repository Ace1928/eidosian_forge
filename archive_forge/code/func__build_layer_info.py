import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def _build_layer_info(self) -> List:
    """
        Helper function to create a list of LayerInfo instances.
        """
    layer_info_list = list()
    for name, layer in self._model.named_modules():
        if name != '':
            if name not in self._layer_scale_dict.keys():
                logging.debug('name = %s, layer = %s, scaling_factor = %s' % (name, layer, 1.0))
                layer_info_list.append(LayerInfo(name, layer, 1.0))
            else:
                logging.debug('name = %s, layer = %s, scaling_factor = %s' % (name, layer, self._layer_scale_dict[name]))
                layer_info_list.append(LayerInfo(name, layer, self._layer_scale_dict[name], True))
    return layer_info_list