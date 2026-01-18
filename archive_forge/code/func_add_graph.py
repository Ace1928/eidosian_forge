from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
def add_graph(self, graph, global_step=None, graph_def=None):
    """Add graph."""
    if global_step is not None and global_step < 0:
        raise ValueError('Invalid global_step %s.' % global_step)
    if graph_def is not None:
        raise ValueError('Unexpected graph_def %s.' % graph_def)
    self._added_graphs.append(graph)