import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _write_graph(self):
    """Writes graph_def to `logdir` and adds it to summary if applicable."""
    assert self._is_chief
    if self._logdir:
        training_util.write_graph(self._graph.as_graph_def(add_shapes=True), self._logdir, 'graph.pbtxt')
    if self._summary_writer and (not self._graph_added_to_summary):
        self._summary_writer.add_graph(self._graph)
        self._summary_writer.add_meta_graph(self._meta_graph_def)
        self._graph_added_to_summary = True