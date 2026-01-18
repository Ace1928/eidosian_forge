import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def _is_valid_filename(filename: str) -> bool:
    """Utility to check whether the provided filename is valid."""
    exp = re.compile('^[^<>:;,?"*|/]+$')
    match = exp.match(filename)
    return bool(match)