import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _generate_outputs(arg_idx, output_list):
    graph = self.optim_node.graph
    with graph.inserting_after(self.optim_node):
        optim_getitem = graph.call_function(operator.getitem, (self.optim_node, arg_idx))
    for i, arg in enumerate(self.optim_node.args[arg_idx]):
        with graph.inserting_after(optim_getitem):
            updated_arg = graph.call_function(operator.getitem, (optim_getitem, i))
        with graph.inserting_after(updated_arg):
            output_copy = graph.call_function(aten.copy_, (arg, updated_arg))
        output_list.append(output_copy)