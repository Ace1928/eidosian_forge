from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def _set_external_data(tensor: onnx.TensorProto, location: str, offset: int | None=None, length: int | None=None, checksum: str | None=None, basepath: str | None=None) -> None:
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for k, v in {'location': location, 'offset': offset, 'length': length, 'checksum': checksum, 'basepath': basepath}.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)