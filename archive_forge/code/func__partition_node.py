from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
@_beartype.beartype
def _partition_node(self, node: torch.Node, complete_upper_nodes_set: Set[torch.Node], complete_lower_nodes_set: Set[torch.Node], original_graph_outputs: Set[torch.Value], covered_bridge_values: Set[torch.Value], process_bridge_value: Callable[[torch.Value], torch.Value]):
    if node in complete_lower_nodes_set:
        return
    if _node_has_uses_by(node, complete_lower_nodes_set) and node.kind() in self._EXCLUDED_NODE_KINDS:
        complete_lower_nodes_set.update(_all_nodes([node]))
        for input in node.inputs():
            if input in covered_bridge_values:
                continue
            self._partition_node(input.node(), complete_upper_nodes_set, complete_lower_nodes_set, original_graph_outputs, covered_bridge_values, process_bridge_value)
    else:
        for output in node.outputs():
            if output in covered_bridge_values:
                continue
            if _has_uses_by_nodes(output, complete_lower_nodes_set) or output in original_graph_outputs:
                covered_bridge_values.add(process_bridge_value(output))