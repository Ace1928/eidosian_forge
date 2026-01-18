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
def _ort_session(model: Union[str, io.BytesIO], ort_providers: Sequence[str]=_ORT_PROVIDERS):
    try:
        import onnxruntime
    except ImportError as e:
        raise ImportError('onnxruntime is required for export verification.') from e
    if ort_providers is None:
        ort_providers = _ORT_PROVIDERS
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(model if isinstance(model, str) else model.getvalue(), session_options, providers=ort_providers)
    return ort_session