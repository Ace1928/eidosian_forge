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
def _unpack_to_numpy(values, cast_onnx_accepted=True) -> list:
    value_unpacked = []
    for value in values:
        value_unpacked.extend(utils.unpack_quantized_tensor(value, cast_onnx_accepted=cast_onnx_accepted))
    return [_to_numpy(v) for v in value_unpacked]