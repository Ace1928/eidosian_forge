import functools
import warnings
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import _type_utils, errors, symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
def _try_cast_integer_to_float(g: jit_utils.GraphContext, *args):
    floating_scalar_types = {_type_utils.JitScalarType.HALF, _type_utils.JitScalarType.FLOAT, _type_utils.JitScalarType.DOUBLE}
    old_type = None
    arg0_type = _type_utils.JitScalarType.from_value(args[0], _type_utils.JitScalarType.UNDEFINED)
    if arg0_type != _type_utils.JitScalarType.UNDEFINED:
        old_type = arg0_type
        if old_type not in floating_scalar_types:
            old_type = old_type.scalar_name()
            args = tuple((g.op('Cast', arg, to_i=_C_onnx.TensorProtoDataType.FLOAT) for arg in args))
        else:
            return (None,) + args
    else:
        warnings.warn('Only floating datatype is supported for these operators: {Greater, Less, MatMul, PRelu, Gemm, Flatten}. This might cause the onnx model to be incorrect, if inputs have integer datatypes.')
    return (old_type,) + args