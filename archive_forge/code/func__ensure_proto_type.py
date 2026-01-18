from __future__ import annotations
import os
import sys
from typing import Any, Callable, TypeVar
from google.protobuf.message import Message
import onnx.defs
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
import onnx.shape_inference
from onnx import (
def _ensure_proto_type(proto: Message, proto_type: type[Message]) -> None:
    if not isinstance(proto, proto_type):
        raise TypeError(f"The proto message needs to be of type '{proto_type.__name__}'")