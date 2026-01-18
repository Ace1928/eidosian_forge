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
def _process_bridge_value_for_lower(graph: torch.Graph, bridge_value: torch.Value) -> torch.Value:
    new_input = graph.addInput()
    bridge_value.replaceAllUsesWith(new_input)
    new_input.copyMetadata(bridge_value)
    return new_input