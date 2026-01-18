from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
def _op_schema_type_constraint_param_repr(self) -> str:
    return f'OpSchema.TypeConstraintParam(type_param_str={self.type_param_str!r}, allowed_type_strs={self.allowed_type_strs!r}, description={self.description!r})'