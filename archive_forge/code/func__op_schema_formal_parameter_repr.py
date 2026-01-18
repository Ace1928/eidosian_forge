from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
def _op_schema_formal_parameter_repr(self) -> str:
    return f'OpSchema.FormalParameter(name={self.name!r}, type_str={self.type_str!r}, description={self.description!r}, param_option={self.option!r}, is_homogeneous={self.is_homogeneous!r}, min_arity={self.min_arity!r}, differentiation_category={self.differentiation_category!r})'