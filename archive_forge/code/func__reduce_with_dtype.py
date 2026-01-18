import functools
import torch
import torch._C._onnx as _C_onnx
from torch.onnx import (
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::sum', decorate=[_apply_params('ReduceSum', 'sum')])
@_beartype.beartype
def _reduce_with_dtype(onnx_op, name):
    symbolic = _reduce_op_symbolic(onnx_op)

    @opset9.overload_by_arg_count
    @_beartype.beartype
    def reduce(g, *args, **kwargs):

        @symbolic_helper.parse_args('v', 'none')
        @_beartype.beartype
        def reduce_nodim(g, self, dtype):
            dtype_onnx = None
            if dtype.node().kind() == 'onnx::Constant':
                dtype = symbolic_helper._get_const(dtype, 'i', 'dtype')
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op('Cast', self, to_i=dtype_onnx)
            elif dtype.node().kind() != 'prim::Constant':
                return symbolic_helper._unimplemented(name, 'dtype', dtype)
            result = symbolic(g, self)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(result).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op('Cast', result, to_i=dtype_onnx)
            return result

        @symbolic_helper.parse_args('v', 'v', 'i', 'none')
        @_beartype.beartype
        def reduce_dim(g, self, dim, keepdim, dtype):
            dtype_onnx = None
            if dtype.node().kind() == 'onnx::Constant':
                dtype = symbolic_helper._get_const(dtype, 'i', 'dtype')
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op('Cast', self, to_i=dtype_onnx)
            elif dtype.node().kind() != 'prim::Constant':
                return symbolic_helper._unimplemented(name, 'dtype', dtype)
            result = symbolic(g, self, dim, keepdim)
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(result).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op('Cast', result, to_i=dtype_onnx)
            return result
        return (reduce_nodim, reduce_dim)
    return reduce