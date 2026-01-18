import collections
import functools
import glob
import os
import tempfile
import threading
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
def _debug_run_and_get_dump(self, sess, fetches, feed_dict=None, debug_ops='DebugIdentity', tolerate_debug_op_creation_failures=False, global_step=-1, validate=True, expected_partition_graph_count=None):
    """Run fetches with debugging and obtain DebugDumpDir.

    Args:
      sess: the tf.compat.v1.Session to be used.
      fetches: fetches of the Session.run().
      feed_dict: feed dict for the Session.run().
      debug_ops: name(s) of the debug ops to be used.
      tolerate_debug_op_creation_failures: whether to tolerate debug op
        creation failures.
      global_step: Optional global step.
      validate: whether to validate dumped tensors against graph.
      expected_partition_graph_count: optional count of partition graphs to
        assert on.

    Returns:
      1. Return values of the Session.run().
      2. The DebugDumpDir object from the debugged run().
    """
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    debug_utils.watch_graph(run_options, sess.graph, debug_ops=debug_ops, debug_urls=self._debug_urls(), tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures, global_step=global_step)
    run_metadata = config_pb2.RunMetadata()
    run_output = sess.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    if expected_partition_graph_count is not None:
        self.assertEqual(expected_partition_graph_count, len(run_metadata.partition_graphs))
    return (run_output, debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=validate))