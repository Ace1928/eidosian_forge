import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence, Tuple
import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose
import onnx.reference.custom_element_types as custom
from onnx import (
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col
@staticmethod
def _cdist_model(opset, reduce_op='ReduceSumSquare'):
    initializers = []
    inputs = [make_tensor_value_info('next_in', TensorProto.FLOAT, [None, 4]), make_tensor_value_info('next', TensorProto.FLOAT, [None])]
    outputs = [make_tensor_value_info('next_out', TensorProto.FLOAT, [None, None]), make_tensor_value_info('scan_out', TensorProto.FLOAT, [None])]
    if opset >= 18:
        initializers.append(from_array(np.array([1], dtype=np.int64), name='axis_red'))
        node_reduce = make_node(reduce_op, ['cdistdf_17_C0', 'axis_red'], ['cdistdf_17_reduced0'], name='cdistdf_17_ReduceSumSquare', keepdims=0)
    else:
        node_reduce = make_node(reduce_op, ['cdistdf_17_C0'], ['cdistdf_17_reduced0'], name='cdistdf_17_ReduceSumSquare', axes=[1], keepdims=0)
    nodes = [make_node('Identity', ['next_in'], ['next_out'], name='cdistd_17_Identity'), make_node('Sub', ['next_in', 'next'], ['cdistdf_17_C0'], name='cdistdf_17_Sub'), node_reduce, make_node('Identity', ['cdistdf_17_reduced0'], ['scan_out'], name='cdistdf_17_Identity')]
    graph = make_graph(nodes, 'OnnxIdentity', inputs, outputs, initializers)
    initializers = []
    list_value = [1.1394007205963135, -0.6848101019859314, -1.234825849533081, 0.4023416340351105, 0.17742614448070526, 0.46278226375579834, -0.4017809331417084, -1.630198359489441, -0.5096521973609924, 0.7774903774261475, -0.4380742907524109, -1.2527953386306763, -1.0485529899597168, 1.950775384902954, -1.420017957687378, -1.7062702178955078, 1.8675580024719238, -0.15135720372200012, -0.9772778749465942, 0.9500884413719177, -2.5529897212982178, -0.7421650290489197, 0.653618574142456, 0.8644362092018127, 1.5327792167663574, 0.37816253304481506, 1.4693588018417358, 0.154947429895401, -0.6724604368209839, -1.7262825965881348, -0.35955315828323364, -0.8131462931632996, -0.8707971572875977, 0.056165341287851334, -0.5788496732711792, -0.3115525245666504, 1.2302906513214111, -0.302302747964859, 1.202379822731018, -0.38732680678367615, 2.269754648208618, -0.18718385696411133, -1.4543657302856445, 0.04575851559638977, -0.9072983860969543, 0.12898291647434235, 0.05194539576768875, 0.7290905714035034, 1.4940791130065918, -0.8540957570075989, -0.2051582634449005, 0.3130677044391632, 1.764052391052246, 2.2408931255340576, 0.40015721321105957, 0.978738009929657, 0.06651721894741058, -0.3627411723136902, 0.30247190594673157, -0.6343221068382263, -0.5108051300048828, 0.4283318817615509, -1.18063223361969, -0.02818222902715206, -1.6138978004455566, 0.38690251111984253, -0.21274028718471527, -0.8954665660858154, 0.7610377073287964, 0.3336743414402008, 0.12167501449584961, 0.44386324286460876, -0.10321885347366333, 1.4542734622955322, 0.4105985164642334, 0.14404356479644775, -0.8877857327461243, 0.15634897351264954, -1.980796456336975, -0.34791216254234314]
    initializers.append(from_array(np.array(list_value, dtype=np.float32).reshape((20, 4)), name='Sc_Scancst'))
    initializers.append(from_array(np.array([2], dtype=np.int64), name='To_TopKcst'))
    inputs = [make_tensor_value_info('input', TensorProto.FLOAT, [None, 4])]
    outputs = [make_tensor_value_info('values', TensorProto.FLOAT, [None, 2]), make_tensor_value_info('indices', TensorProto.INT64, [None, 2])]
    nodes = [make_node('Scan', ['input', 'Sc_Scancst'], ['UU032UU', 'UU033UU'], name='Sc_Scan', body=graph, num_scan_inputs=1), make_node('Transpose', ['UU033UU'], ['Tr_transposed0'], name='Tr_Transpose', perm=[1, 0]), make_node('Sqrt', ['Tr_transposed0'], ['Sq_Y0'], name='Sq_Sqrt'), make_node('TopK', ['Sq_Y0', 'To_TopKcst'], ['values', 'indices'], name='To_TopK', largest=0, sorted=1)]
    graph = make_graph(nodes, 'dummy', inputs, outputs, initializers)
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', opset)])
    return onnx_model