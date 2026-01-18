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
def _get_test_tree_ensemble_opset_latest(aggregate_function, rule=Mode.LEQ, unique_targets=False, input_type=TensorProto.FLOAT):
    X = make_tensor_value_info('X', input_type, [None, None])
    Y = make_tensor_value_info('Y', input_type, [None, None])
    if unique_targets:
        weights = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    else:
        weights = [0.07692307978868484, 0.5, 0.5, 0.0, 0.2857142984867096, 0.5]
    node = make_node('TreeEnsemble', ['X'], ['Y'], domain='ai.onnx.ml', n_targets=1, aggregate_function=aggregate_function, membership_values=None, nodes_missing_value_tracks_true=None, nodes_hitrates=None, post_transform=0, tree_roots=[0, 2], nodes_splits=make_tensor('node_splits', input_type, (4,), [0.26645058393478394, 0.6214364767074585, -0.5592705607414246, -0.7208403944969177]), nodes_featureids=[0, 2, 0, 0], nodes_modes=make_tensor('nodes_modes', TensorProto.UINT8, (4,), [rule] * 4), nodes_truenodeids=[1, 0, 3, 4], nodes_trueleafs=[0, 1, 1, 1], nodes_falsenodeids=[2, 1, 3, 5], nodes_falseleafs=[1, 1, 0, 1], leaf_targetids=[0, 0, 0, 0, 0, 0], leaf_weights=make_tensor('leaf_weights', input_type, (len(weights),), weights))
    graph = make_graph([node], 'ml', [X], [Y])
    model = make_model_gen_version(graph, opset_imports=OPSETS)
    return model