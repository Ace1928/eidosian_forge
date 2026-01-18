import collections
import dataclasses
import re
import sys
import types
from typing import Counter, Dict, List, Optional
import torch.nn
from . import utils
from .bytecode_transformation import (
from .exc import unimplemented
from .source import AttrSource, Source
from .utils import is_safe_constant, rot_n_helper
from .variables.base import VariableTracker
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
def add_graph_output(self, value):
    graph_outputs_key = id(value.as_proxy())
    if graph_outputs_key not in self.graph_outputs:
        self.graph_outputs[graph_outputs_key] = GraphOutputEntry(len(self.graph_outputs), value)
    return graph_outputs_key