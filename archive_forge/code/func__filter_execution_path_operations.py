import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def _filter_execution_path_operations(self, operations, fetches):
    """Returns the set of ops in the execution path to compute given fetches."""
    if fetches is None:
        return set(operations)
    if not isinstance(fetches, (list, tuple)):
        fetches = [fetches]
    op_fetches = []
    for fetch in fetches:
        if isinstance(fetch, ops.Operation):
            op_fetches.append(fetch)
        elif isinstance(fetch, tensor_lib.Tensor):
            op_fetches.append(fetch.op)
        else:
            raise RuntimeError('Given fetch:%s is neither a tensor nor an op.' % fetch)
    execution_path_operations = set(op_fetches)
    traverse_stack = list(op_fetches)
    while True:
        if not traverse_stack:
            break
        head_op = traverse_stack.pop()
        input_ops = [tensor_input.op for tensor_input in head_op.inputs]
        input_ops.extend(head_op.control_inputs)
        for input_op in input_ops:
            if input_op not in execution_path_operations:
                if TensorTracer.loop_cond_op(input_op):
                    continue
                execution_path_operations.add(input_op)
                traverse_stack.append(input_op)
    return execution_path_operations