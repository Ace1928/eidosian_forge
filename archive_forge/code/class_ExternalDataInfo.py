import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
class ExternalDataInfo:

    def __init__(self, tensor: TensorProto) -> None:
        self.location = ''
        self.offset = None
        self.length = None
        self.checksum = None
        self.basepath = ''
        for entry in tensor.external_data:
            setattr(self, entry.key, entry.value)
        if self.offset:
            self.offset = int(self.offset)
        if self.length:
            self.length = int(self.length)