from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def __set_state__(self, state: Dict[str, Any]) -> None:
    state['state'] = self._convert_mask(state['state'], sparse_coo=False)
    self.__dict__.update(state)
    for name, config in self.data_groups.items():
        layer = fqn_to_module(self.model, name)
        assert layer is not None
        if 'hook_state' in config and config['hook_state'] == 'aggregate':
            hook = layer.register_forward_pre_hook(self._aggregate_hook(name))
        elif 'hook_state' in config and config['hook_state'] == 'sparsify':
            hook = layer.register_forward_pre_hook(self._sparsify_hook(name))
        config['layer'] = layer
        config['hook'] = hook