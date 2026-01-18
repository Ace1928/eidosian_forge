import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('SparseMatMul')
def _SparseMatMulGrad(op, grad):
    """Gradient for SparseMatMul."""
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    is_sparse = {}
    is_sparse[op.inputs[0].ref()] = op.get_attr('a_is_sparse')
    is_sparse[op.inputs[1].ref()] = op.get_attr('b_is_sparse')
    is_sparse[grad.ref()] = not context.executing_eagerly() and grad.op.type == 'ReluGrad'

    def _SparseMatMul(t1, t2, out_dtype, transpose_a=False, transpose_b=False):
        """Helper function to create SparseMatMul op."""
        assert t1.ref() in is_sparse and t2.ref() in is_sparse
        t1_sparse = is_sparse[t1.ref()]
        t2_sparse = is_sparse[t2.ref()]
        if transpose_b:
            t2 = array_ops.transpose(t2)
            transpose_b = False
        prod = math_ops.matmul(t1, t2, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=t1_sparse, b_is_sparse=t2_sparse)
        if prod.dtype != out_dtype:
            prod = math_ops.cast(prod, out_dtype)
        return prod
    dtype_a = op.inputs[0].dtype
    dtype_b = op.inputs[1].dtype
    if not t_a and (not t_b):
        return (_SparseMatMul(grad, op.inputs[1], dtype_a, transpose_b=True), _SparseMatMul(op.inputs[0], grad, dtype_b, transpose_a=True))
    elif not t_a and t_b:
        return (_SparseMatMul(grad, op.inputs[1], dtype_a), _SparseMatMul(grad, op.inputs[0], dtype_b, transpose_a=True))
    elif t_a and (not t_b):
        return (_SparseMatMul(op.inputs[1], grad, dtype_a, transpose_b=True), _SparseMatMul(op.inputs[0], grad, dtype_b))
    elif t_a and t_b:
        return (_SparseMatMul(op.inputs[1], grad, dtype_a, transpose_a=True, transpose_b=True), _SparseMatMul(grad, op.inputs[0], dtype_b, transpose_a=True, transpose_b=True))