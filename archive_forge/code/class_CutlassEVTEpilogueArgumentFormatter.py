from typing import Dict, List
from unittest.mock import patch
import sympy
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.utils import IndentedBuffer, sympy_str
class CutlassEVTEpilogueArgumentFormatter:
    """
    Codegen class, which provides an entry point to generate
    Cutlass "Epilogue Visitor Tree" (EVT) Argument initializers

    See https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder
    for more about EVTs and how they are declared and used to generate.

    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.


    """

    def __init__(self, accumulator_node_name: str):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string static method.

        Args:
            accumulator_node_name (str): The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
        """
        self.accumulator_node_name: str = accumulator_node_name
        self.output: IndentedBuffer = IndentedBuffer(0)
        self.var_counter: int = 0
        self.aliases: Dict[str, str] = dict()

    @staticmethod
    def ir_to_evt_argument_string(template_output_node_name: str, epilogue_nodes: List[IRNode]) -> str:
        formatter = CutlassEVTEpilogueArgumentFormatter(template_output_node_name)
        with virtualized.V.set_ops_handler(formatter), patch.object(FlexibleLayout, 'allow_indexing', True):
            for node in epilogue_nodes:
                assert isinstance(node, ComputedBuffer)
                pnode = node.data
                assert isinstance(pnode, Pointwise)
                index = pnode._index(pnode.ranges)
                result = pnode.inner_fn(index)
                if node.name is not None:
                    formatter.aliases[node.name] = result
            res: str = formatter.getvalue(result)
            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise CUTLASSEVTOpNotImplementedError('sympy / indexing expressions not yet supported in EVT fusion')
            else:
                return res

    def __getattr__(self, name):

        def inner(*args, **kwargs):
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for key, a in kwargs.items()}
            fn = getattr(self, f'_op_{name}')
            line = fn(*fargs, **fkwargs)
            return line
        if name.startswith('_'):
            raise CUTLASSEVTOpNotImplementedError(name)
        if hasattr(self, f'_op_{name}'):
            return inner
        else:
            raise CUTLASSEVTOpNotImplementedError(name)

    def _op_load(self, name, index_expr):
        if name == self.accumulator_node_name:
            return '{}'
        elif name in self.aliases:
            return self.aliases[name]
        else:
            raise CUTLASSEVTOpNotImplementedError(f'Operand {name} not found. Auxiliary inputs not supported yet.')

    def _op_constant(self, value, dtype):
        if str(dtype) in ('torch.float16', 'torch.float32'):
            return '{ static_cast<ElementAcc>(' + str(value) + ') }'
        else:
            raise CUTLASSEVTOpNotImplementedError(f'Unsupported dtype for constant: {dtype}')

    def _cutlass_binary_functional_op(self, op, a, b):
        return f'{{ /*{op}: */ {a}, {b} }}'

    def _op_mul(self, a, b):
        return self._cutlass_binary_functional_op('multiplies', a, b)

    def _op_div(self, a, b):
        return self._cutlass_binary_functional_op('divides', a, b)

    def _op_truediv(self, a, b):
        return self._cutlass_binary_functional_op('divides', a, b)

    def _op_ge(self, a, b):
        return self._cutlass_binary_functional_op('greater_equal', a, b)

    def _op_add(self, a, b):
        return self._cutlass_binary_functional_op('plus', a, b)

    def _op_sub(self, a, b):
        return self._cutlass_binary_functional_op('minus', a, b)

    def _op_minimum(self, a, b):
        return self._cutlass_binary_functional_op('minimum', a, b)

    def _op_maximum(self, a, b):
        return self._cutlass_binary_functional_op('maximum', a, b)

    def _op_relu(self, a):
        const_zero = self._op_constant(0.0, 'torch.float32')
        return '{' + str(a) + ', ' + const_zero + '}'

    def _op_to_dtype(self, a, dtype, src_dtype=None):
        assert dtype in ('torch.float32', 'torch.float16'), f'Unsupported dtype: {dtype}'
        assert src_dtype in (None, 'torch.float32', 'torch.float16'), f'Unsupported source dtype: {src_dtype}'
        return a

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise CUTLASSEVTOpNotImplementedError()

    def getvalue(self, result) -> str:
        return '{' + str(result) + '}'