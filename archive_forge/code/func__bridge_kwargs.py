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
def _bridge_kwargs(self):
    pt_outs = self.pt_outs
    graph_outputs = list(self.graph.outputs())
    assert pt_outs is not None
    assert len(graph_outputs) == len(pt_outs), f'{len(graph_outputs)} vs {len(pt_outs)}\nGraph: {self.graph}'
    return {v.debugName(): o for v, o in zip(graph_outputs, pt_outs)}