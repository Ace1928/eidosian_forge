import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def _test_overlapping_names(self, inputs0: Sequence[str]=('i0', 'i1'), inputs1: Sequence[str]=('i2', 'i3'), outputs0: Sequence[str]=('o0', 'o1'), outputs1: Sequence[str]=('o2', 'o3'), value_info0: Sequence[str]=('v0', 'v1'), value_info1: Sequence[str]=('v2', 'v3'), initializer0: Sequence[str]=('init0', 'init1'), initializer1: Sequence[str]=('init2', 'init3'), sparse_initializer0: Sequence[str]=('sparse_init0', 'sparse_init1'), sparse_initializer1: Sequence[str]=('sparse_init2', 'sparse_init3')) -> None:
    n0 = [helper.make_node('Identity', inputs=[inputs0[i]], outputs=[outputs0[i]]) for i in range(len(inputs0))]
    i0 = [helper.make_tensor_value_info(inputs0[i], TensorProto.FLOAT, []) for i in range(len(inputs0))]
    o0 = [helper.make_tensor_value_info(outputs0[i], TensorProto.FLOAT, []) for i in range(len(outputs0))]
    vi0 = [helper.make_tensor_value_info(value_info0[i], TensorProto.FLOAT, []) for i in range(len(value_info0))]
    init0 = [helper.make_tensor(name=initializer0[i], data_type=TensorProto.INT64, dims=(), vals=[1]) for i in range(len(initializer0))]
    sparse_init0 = [_make_sparse_tensor(sparse_initializer0[i]) for i in range(len(sparse_initializer0))]
    n1 = [helper.make_node('Identity', inputs=[inputs1[i]], outputs=[outputs1[i]]) for i in range(len(inputs1))]
    i1 = [helper.make_tensor_value_info(inputs1[i], TensorProto.FLOAT, []) for i in range(len(inputs1))]
    o1 = [helper.make_tensor_value_info(outputs1[i], TensorProto.FLOAT, []) for i in range(len(outputs1))]
    vi1 = [helper.make_tensor_value_info(value_info1[i], TensorProto.FLOAT, []) for i in range(len(value_info1))]
    init1 = [helper.make_tensor(name=initializer1[i], data_type=TensorProto.INT64, dims=(), vals=[1]) for i in range(len(initializer1))]
    sparse_init1 = [_make_sparse_tensor(sparse_initializer1[i]) for i in range(len(sparse_initializer1))]
    ops = [helper.make_opsetid('', 10)]
    m0 = helper.make_model(helper.make_graph(nodes=n0, name='g0', inputs=i0, outputs=o0, value_info=vi0, initializer=init0, sparse_initializer=sparse_init0), producer_name='test', opset_imports=ops)
    m1 = helper.make_model(helper.make_graph(nodes=n1, name='g1', inputs=i1, outputs=o1, value_info=vi1, initializer=init1, sparse_initializer=sparse_init1), producer_name='test', opset_imports=ops)
    overlap = compose.check_overlapping_names(m0.graph, m1.graph)
    i = 0
    overlapping_inputs = list(set(inputs0) & set(inputs1))
    overlapping_outputs = list(set(outputs0) & set(outputs1))
    overlapping_edges = list(set(overlapping_inputs + overlapping_outputs))
    if overlapping_edges:
        self.assertEqual(overlap[i], ('edge', overlapping_edges))
        i += 1
    overlapping_vis = list(set(value_info0) & set(value_info1))
    if overlapping_vis:
        self.assertEqual(overlap[i], ('value_info', overlapping_vis))
        i += 1
    overlapping_init = list(set(initializer0) & set(initializer1))
    if overlapping_init:
        self.assertEqual(overlap[i], ('initializer', overlapping_init))
        i += 1
    overlapping_sparse_init = list(set(sparse_initializer0) & set(sparse_initializer1))
    if overlapping_sparse_init:
        expected_overlap = []
        for overlapping_name in overlapping_sparse_init:
            expected_overlap.append(overlapping_name + '_values')
            expected_overlap.append(overlapping_name + '_idx')
        self.assertEqual(overlap[i], ('sparse_initializer', expected_overlap))
        i += 1
    m0_new = compose.add_prefix(m0, prefix='g0/')
    overlap = compose.check_overlapping_names(m0_new.graph, m1.graph)
    self.assertEqual(0, len(overlap))