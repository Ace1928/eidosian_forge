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
def _get_test_tree_ensemble_regressor(aggregate_function, rule='BRANCH_LEQ', unique_targets=False, base_values=None):
    opsets = [make_opsetid('', TARGET_OPSET), make_opsetid('ai.onnx.ml', 3)]
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    if unique_targets:
        targets = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    else:
        targets = [0.07692307978868484, 0.5, 0.5, 0.0, 0.2857142984867096, 0.5]
    node1 = make_node('TreeEnsembleRegressor', ['X'], ['Y'], domain='ai.onnx.ml', n_targets=1, aggregate_function=aggregate_function, base_values=base_values, nodes_falsenodeids=[4, 3, 0, 0, 0, 2, 0, 4, 0, 0], nodes_featureids=[0, 2, 0, 0, 0, 0, 0, 2, 0, 0], nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], nodes_modes=[rule, rule, 'LEAF', 'LEAF', 'LEAF', rule, 'LEAF', rule, 'LEAF', 'LEAF'], nodes_nodeids=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], nodes_treeids=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], nodes_truenodeids=[1, 2, 0, 0, 0, 1, 0, 3, 0, 0], nodes_values=[0.26645058393478394, 0.6214364767074585, 0.0, 0.0, 0.0, -0.7208403944969177, 0.0, -0.5592705607414246, 0.0, 0.0], post_transform='NONE', target_ids=[0, 0, 0, 0, 0, 0], target_nodeids=[2, 3, 4, 1, 3, 4], target_treeids=[0, 0, 0, 1, 1, 1], target_weights=targets)
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=opsets)
    check_model(onx)
    return onx