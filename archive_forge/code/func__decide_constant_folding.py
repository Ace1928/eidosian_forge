from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _decide_constant_folding(do_constant_folding, operator_export_type, training):
    do_constant_folding = _resolve_args_by_export_type('do_constant_folding', do_constant_folding, operator_export_type)
    if do_constant_folding and (training is not None and training is not _C_onnx.TrainingMode.EVAL):
        warnings.warn("It is recommended that constant folding be turned off ('do_constant_folding=False') when exporting the model in training-amenable mode, i.e. with 'training=TrainingMode.TRAIN' or 'training=TrainingMode.PRESERVE' (when model is in training mode). Otherwise, some learnable model parameters may not translate correctly in the exported ONNX model because constant folding mutates model parameters. Please consider turning off constant folding or setting the training=TrainingMode.EVAL.")
    return do_constant_folding