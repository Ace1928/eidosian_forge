import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
@dataclasses.dataclass
class GraphInputMatcher:
    """
    The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
    Specifically, those graph inputs corresponding to method parameters should be replaced with the
    arguments for the current call.

    tensor_id_to_arg_idx maps the tensor id to the parameter index.
    graph_input_tensor_ids, graph_input_ivalues list the tensor_id and ivalue for each of the
    TS/XLA graph inputs.
    """
    tensor_id_to_arg_idx: Dict[int, int]
    graph_input_tensor_ids: List[int]
    graph_input_ivalues: List[Any]

    def __call__(self, args):
        real_input = []
        for tensor_id, traced_ivalue in zip(self.graph_input_tensor_ids, self.graph_input_ivalues):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
            if arg_idx is None:
                inp = traced_ivalue
            else:
                inp = args[arg_idx]
            real_input.append(inp)
        return real_input