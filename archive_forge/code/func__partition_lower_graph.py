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
def _partition_lower_graph(self) -> torch.Graph:
    pivot = self._graph_partition_pivot()
    if pivot == -1:
        return torch.Graph()
    graph = self.graph.copy()
    original_outputs = list(graph.outputs())
    original_inputs = list(graph.inputs())
    new_outputs = []

    def _process_bridge_value_for_lower(graph: torch.Graph, bridge_value: torch.Value) -> torch.Value:
        new_input = graph.addInput()
        bridge_value.replaceAllUsesWith(new_input)
        new_input.copyMetadata(bridge_value)
        return new_input
    process_bridge_value_for_lower = functools.partial(_process_bridge_value_for_lower, graph)
    upper_nodes, lower_nodes, _, complete_lower_nodes_set = self._partition_nodes(graph, pivot, process_bridge_value_for_lower)
    for output in original_outputs:
        if _produced_by(output, lower_nodes):
            new_outputs.append(output)
    for _ in enumerate(original_outputs):
        graph.eraseOutput(0)
    for output in new_outputs:
        graph.registerOutput(output)
    for input in original_inputs:
        if _has_uses_by_nodes(input, complete_lower_nodes_set):
            new_input = graph.addInput()
            input.replaceAllUsesWith(new_input)
            new_input.copyMetadata(input)
    for node in reversed(upper_nodes):
        if node not in complete_lower_nodes_set:
            try:
                node.destroy()
            except RuntimeError as e:
                print(node, graph)
                raise e
    for _ in original_inputs:
        graph.eraseInput(0)
    return graph