import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.reference as orf
def _make_local_graph_body():
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    sparse_initializers = []
    inputs.append(oh.make_tensor_value_info('infinite_loop', onnx.TensorProto.INT64, shape=[]))
    inputs.append(oh.make_tensor_value_info('cond', onnx.TensorProto.BOOL, shape=[]))
    inputs.append(oh.make_tensor_value_info('T', onnx.TensorProto.UNDEFINED, []))
    nodes.append(oh.make_node('Add', ['T', 'A'], ['T_0']))
    nodes.append(oh.make_node('ReduceSum', ['T_0'], ['tmp']))
    nodes.append(oh.make_node('Constant', [], ['int64_m10'], value=onh.from_array(np.array(-10, dtype=np.int64), name='value')))
    nodes.append(oh.make_node('CastLike', ['int64_m10', 'tmp'], ['int64_m10_cast']))
    nodes.append(oh.make_node('Greater', ['tmp', 'int64_m10_cast'], ['cond_1']))
    nodes.append(oh.make_node('Identity', ['cond_1'], ['cond_out']))
    outputs.append(oh.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, shape=[]))
    outputs.append(oh.make_tensor_value_info('T_0', onnx.TensorProto.UNDEFINED, []))
    graph = oh.make_graph(nodes, 'loop_body', inputs, outputs, initializers, sparse_initializer=sparse_initializers)
    return graph