import itertools
import unittest
from functools import wraps
from os import getenv
import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore
from parameterized import parameterized
import onnx
from onnx import ONNX_ML, TensorProto, TypeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_ml_opset_version, onnx_opset_version
from onnx.helper import (
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.aionnxml.op_tree_ensemble import (
@staticmethod
def _get_test_svm_classifier_linear_sv(post_transform, probability=True):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    In = make_tensor_value_info('I', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    kwargs = {'classlabels_ints': [0, 1], 'coefficients': [0.766398549079895, 0.0871576070785522, 0.110420741140842, -0.963976919651031], 'support_vectors': [4.80000019073486, 3.40000009536743, 1.89999997615814, 5.0, 3.0, 1.60000002384186, 4.5, 2.29999995231628, 1.29999995231628, 5.09999990463257, 2.5, 3.0], 'kernel_params': [0.122462183237076, 0.0, 3.0], 'kernel_type': 'LINEAR', 'prob_a': [-5.139118194580078], 'prob_b': [0.06399919837713242], 'rho': [2.23510527610779], 'post_transform': post_transform, 'vectors_per_class': [3, 1]}
    if not probability:
        del kwargs['prob_a']
        del kwargs['prob_b']
    node1 = make_node('SVMClassifier', ['X'], ['I', 'Y'], domain='ai.onnx.ml', **kwargs)
    graph = make_graph([node1], 'ml', [X], [In, Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    return onx