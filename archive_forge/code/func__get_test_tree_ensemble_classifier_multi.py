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
def _get_test_tree_ensemble_classifier_multi(post_transform):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    In = make_tensor_value_info('I', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('TreeEnsembleClassifier', ['X'], ['I', 'Y'], domain='ai.onnx.ml', class_ids=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], class_nodeids=[2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 3, 3, 3, 4, 4, 4], class_treeids=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], class_weights=[0.46666666865348816, 0.0, 0.03333333507180214, 0.20000000298023224, 0.23999999463558197, 0.05999999865889549, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.44999998807907104, 0.0, 0.05000000074505806, 0.10294117778539658, 0.19117647409439087, 0.20588235557079315], classlabels_int64s=[0, 1, 2], nodes_falsenodeids=[4, 3, 0, 0, 0, 2, 0, 4, 0, 0], nodes_featureids=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0], nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], nodes_modes=['BRANCH_LEQ', 'BRANCH_LEQ', 'LEAF', 'LEAF', 'LEAF', 'BRANCH_LEQ', 'LEAF', 'BRANCH_LEQ', 'LEAF', 'LEAF'], nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], nodes_truenodeids=[1, 2, 0, 0, 0, 1, 0, 3, 0, 0], nodes_values=[1.2495747804641724, -0.3050493597984314, 0.0, 0.0, 0.0, -1.6830512285232544, 0.0, -0.6751254796981812, 0.0, 0.0], post_transform=post_transform)
    graph = make_graph([node1], 'ml', [X], [In, Y])
    onx = make_model_gen_version(graph, opset_imports=[make_opsetid('', TARGET_OPSET), make_opsetid('ai.onnx.ml', 3)])
    check_model(onx)
    return onx