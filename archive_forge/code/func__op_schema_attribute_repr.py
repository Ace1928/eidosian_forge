from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
def _op_schema_attribute_repr(self) -> str:
    return f'OpSchema.Attribute(name={self.name!r}, type={self.type!r}, description={self.description!r}, default_value={self.default_value!r}, required={self.required!r})'