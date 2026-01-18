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
def _check_graph_diff(model: Union[torch.nn.Module, torch.jit.ScriptModule], test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]], export_options: _experimental.ExportOptions, model_to_graph_func: Callable[[torch.nn.Module, Tuple[Any, ...], Mapping[str, Any], _experimental.ExportOptions], _C.Graph]) -> str:
    """Check if graph produced by `model_to_graph_func` is the same across `test_input_groups`.

    Args:
        model: See :func:`check_export_model_diff`.
        test_input_groups: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.
        model_to_graph_func: A function to convert a PyTorch model to a JIT IR graph.

    Returns:
        graph_diff_report (str): A string representation of the graph difference.
    """
    if len(test_input_groups) < 2:
        raise ValueError('Need at least two groups of test inputs to compare.')
    ref_jit_graph = None
    for args, kwargs in test_input_groups:
        jit_graph = model_to_graph_func(model, args, kwargs, export_options)
        if ref_jit_graph is None:
            ref_jit_graph = jit_graph
            continue
        graph_diff_report = _GraphDiff(ref_jit_graph, jit_graph).diff_report()
        if graph_diff_report:
            return graph_diff_report
    return ''