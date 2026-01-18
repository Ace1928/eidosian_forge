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
def _get_test_svm_classifier_binary(post_transform, probability=True, linear=False):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    In = make_tensor_value_info('I', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    if linear:
        kwargs = {'classlabels_ints': [0, 1, 2, 3], 'coefficients': [-0.155181212, 0.242698956, 0.00701893432, 0.407614474, -0.0324927823, 0.000279897536, -0.195771302, -0.352437368, -0.0215973096, -0.438190277, 0.0456869105, -0.0129375499], 'kernel_params': [0.001, 0.0, 3.0], 'kernel_type': 'LINEAR', 'prob_a': [-5.139118194580078], 'prob_b': [0.06399919837713242], 'rho': [-0.07489691, -0.1764396, -0.21167431, -0.51619097], 'post_transform': post_transform}
    else:
        kwargs = {'classlabels_ints': [0, 1], 'coefficients': [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], 'kernel_params': [0.3824487328529358, 0.0, 3.0], 'kernel_type': 'RBF', 'prob_a': [-5.139118194580078], 'prob_b': [0.06399919837713242], 'rho': [0.16708599030971527], 'support_vectors': [0.19125767052173615, -1.062204122543335, 0.5006636381149292, -0.5892484784126282, -0.3196830451488495, 0.0984845906496048, 0.24746321141719818, -1.1535362005233765, 0.4109955430030823, -0.5937694907188416, -1.3183348178863525, -1.6423596143722534, 0.558641254901886, -0.9218668341636658, 0.6264089345932007, -0.16060839593410492, -0.6365169882774353, 0.8335472345352173, 0.7539799213409424, -0.3970031440258026, -0.1780400276184082, -0.616622805595398, 0.49261474609375, 0.4470972716808319], 'vectors_per_class': [4, 4], 'post_transform': post_transform}
    if not probability:
        del kwargs['prob_a']
        del kwargs['prob_b']
    node1 = make_node('SVMClassifier', ['X'], ['I', 'Y'], domain='ai.onnx.ml', **kwargs)
    graph = make_graph([node1], 'ml', [X], [In, Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    return onx