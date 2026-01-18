import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
@RegisterPFor('Abs')
@RegisterPFor('Acos')
@RegisterPFor('Acosh')
@RegisterPFor('Add')
@RegisterPFor('AddV2')
@RegisterPFor('Angle')
@RegisterPFor('Asin')
@RegisterPFor('Asinh')
@RegisterPFor('Atan')
@RegisterPFor('Atan2')
@RegisterPFor('Atanh')
@RegisterPFor('BesselI0')
@RegisterPFor('BesselI1')
@RegisterPFor('BesselI0e')
@RegisterPFor('BesselI1e')
@RegisterPFor('BesselK0')
@RegisterPFor('BesselK1')
@RegisterPFor('BesselK0e')
@RegisterPFor('BesselK1e')
@RegisterPFor('BesselJ0')
@RegisterPFor('BesselJ1')
@RegisterPFor('BesselY0')
@RegisterPFor('BesselY1')
@RegisterPFor('BitwiseAnd')
@RegisterPFor('BitwiseOr')
@RegisterPFor('BitwiseXor')
@RegisterPFor('Ceil')
@RegisterPFor('Complex')
@RegisterPFor('ComplexAbs')
@RegisterPFor('Conj')
@RegisterPFor('Cos')
@RegisterPFor('Cosh')
@RegisterPFor('Dawsn')
@RegisterPFor('Digamma')
@RegisterPFor('Div')
@RegisterPFor('DivNoNan')
@RegisterPFor('Elu')
@RegisterPFor('Erf')
@RegisterPFor('Erfc')
@RegisterPFor('Erfinv')
@RegisterPFor('Exp')
@RegisterPFor('Expint')
@RegisterPFor('Expm1')
@RegisterPFor('Floor')
@RegisterPFor('FloorDiv')
@RegisterPFor('FloorMod')
@RegisterPFor('FresnelCos')
@RegisterPFor('FresnelSin')
@RegisterPFor('Greater')
@RegisterPFor('GreaterEqual')
@RegisterPFor('Igamma')
@RegisterPFor('IgammaGradA')
@RegisterPFor('Igammac')
@RegisterPFor('Imag')
@RegisterPFor('Inv')
@RegisterPFor('Invert')
@RegisterPFor('IsFinite')
@RegisterPFor('IsInf')
@RegisterPFor('IsNan')
@RegisterPFor('LeftShift')
@RegisterPFor('Less')
@RegisterPFor('LessEqual')
@RegisterPFor('Lgamma')
@RegisterPFor('Log')
@RegisterPFor('Log1p')
@RegisterPFor('LogicalAnd')
@RegisterPFor('LogicalNot')
@RegisterPFor('LogicalOr')
@RegisterPFor('LogicalXor')
@RegisterPFor('Maximum')
@RegisterPFor('Minimum')
@RegisterPFor('Mod')
@RegisterPFor('Mul')
@RegisterPFor('MulNoNan')
@RegisterPFor('Ndtri')
@RegisterPFor('Neg')
@RegisterPFor('Polygamma')
@RegisterPFor('Pow')
@RegisterPFor('Real')
@RegisterPFor('RealDiv')
@RegisterPFor('Reciprocal')
@RegisterPFor('Relu')
@RegisterPFor('Relu6')
@RegisterPFor('RightShift')
@RegisterPFor('Rint')
@RegisterPFor('Round')
@RegisterPFor('Rsqrt')
@RegisterPFor('Selu')
@RegisterPFor('Sigmoid')
@RegisterPFor('Sign')
@RegisterPFor('Sin')
@RegisterPFor('Sinh')
@RegisterPFor('Softplus')
@RegisterPFor('Softsign')
@RegisterPFor('Spence')
@RegisterPFor('Sqrt')
@RegisterPFor('Square')
@RegisterPFor('SquaredDifference')
@RegisterPFor('Sub')
@RegisterPFor('Tan')
@RegisterPFor('Tanh')
@RegisterPFor('TruncateDiv')
@RegisterPFor('TruncateMod')
@RegisterPFor('Xdivy')
@RegisterPFor('Xlogy')
@RegisterPFor('Xlog1py')
@RegisterPFor('Zeta')
def _convert_cwise(pfor_input):
    if pfor_input.num_inputs > 1:
        pfor_input.expanddim_inputs_for_broadcast()
    out = _create_op(pfor_input.op_type, [x.t for x in pfor_input.inputs], [x.dtype for x in pfor_input.outputs], attrs=pfor_input.op.node_def.attr).outputs
    assert len(out) == 1
    out = out[0]
    op_output = wrap(out, True)
    return op_output