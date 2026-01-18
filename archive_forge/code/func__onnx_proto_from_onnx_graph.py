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
def _onnx_proto_from_onnx_graph(onnx_graph: torch.Graph, export_options: _experimental.ExportOptions, params_dict: Dict[str, Any]) -> Tuple[bytes, Mapping[str, bytes]]:
    opset_version = export_options.opset_version or _constants.ONNX_DEFAULT_OPSET
    dynamic_axes = export_options.dynamic_axes or {}
    operator_export_type = export_options.operator_export_type
    val_keep_init_as_ip = utils._decide_keep_init_as_input(export_options.keep_initializers_as_inputs, operator_export_type, opset_version)
    val_add_node_names = utils._decide_add_node_names(True, operator_export_type)
    custom_opsets = export_options.custom_opsets or {}
    proto, export_map, _, _ = onnx_graph._export_onnx(params_dict, opset_version, dynamic_axes, False, operator_export_type, not export_options.verbose, val_keep_init_as_ip, custom_opsets, val_add_node_names, '', {})
    return (proto, export_map)