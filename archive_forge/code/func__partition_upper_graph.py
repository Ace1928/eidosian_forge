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
def _partition_upper_graph(self) -> torch.Graph:
    pivot = self._graph_partition_pivot()
    if pivot == -1:
        return torch.Graph()
    graph = self.graph.copy()
    original_outputs = list(graph.outputs())

    def _process_bridge_value_for_upper(new_outputs: List[torch.Value], bridge_value: torch.Value) -> torch.Value:
        new_outputs.append(bridge_value)
        return bridge_value
    new_outputs: List[torch.Value] = []
    process_bridge_value_for_upper = functools.partial(_process_bridge_value_for_upper, new_outputs)
    _, dropped_nodes, complete_upper_nodes_set, _ = self._partition_nodes(graph, pivot, process_bridge_value_for_upper)
    for _ in enumerate(original_outputs):
        graph.eraseOutput(0)
    for output in new_outputs:
        graph.registerOutput(output)
    for node in reversed(dropped_nodes):
        node.destroy()
    for i, input in reversed(list(enumerate(list(graph.inputs())))):
        if not _has_uses_by_nodes(input, complete_upper_nodes_set) and input not in new_outputs:
            try:
                graph.eraseInput(i)
            except RuntimeError as e:
                print(input, graph)
                raise e
    return graph