from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
@property
def _attribute_default_value(self):
    attr = AttributeProto()
    attr.ParseFromString(self._default_value)
    return attr