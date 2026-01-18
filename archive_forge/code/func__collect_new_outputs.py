from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def _collect_new_outputs(self, names: list[str]) -> list[ValueInfoProto]:
    return self._collect_new_io_core(self.graph.output, names)