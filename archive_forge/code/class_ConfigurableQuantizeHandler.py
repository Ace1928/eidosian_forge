from abc import ABC
from typing import Callable, Dict, List, Optional, Type
import torch
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.utils import NodePattern, Pattern, QuantizerCls
from torch.fx.graph import Node
from .utils import all_node_args_have_no_tensors
class ConfigurableQuantizeHandler(QuantizeHandler):

    def __init__(self, node_pattern: NodePattern, modules: Dict[str, torch.nn.Module], root_node_getter: Optional[Callable]=None):
        super().__init__(node_pattern, modules, root_node_getter)
        if num_tensor_args_to_observation_type:
            assert self.num_tensor_args in num_tensor_args_to_observation_type, f'Must provide observation_type config for tensor number {self.num_tensor_args} in num_tensor_args_to_observation_type for {node_pattern}'
            self.observation_type = num_tensor_args_to_observation_type[self.num_tensor_args]
        else:
            self.observation_type = observation_type
        self.dtype_configs = dtype_configs

    def is_general_tensor_value_op(self) -> bool:
        return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT