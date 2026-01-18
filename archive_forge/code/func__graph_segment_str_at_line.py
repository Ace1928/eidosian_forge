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
def _graph_segment_str_at_line(self, line: int) -> str:
    """Get the string representation of the graph segment at the given line."""
    if line == 0:
        result_str = self._node_count_segment_str()
        result_str += ' ' * (self._max_segment_columns() - len(result_str))
        return result_str
    if line == 1:
        result_str = self._graph_id_segment_str()
        result_str += ' ' * (self._max_segment_columns() - len(result_str))
        return result_str
    if 0 <= line < self._total_rows():
        return ' ' * self._max_segment_columns()
    return ''