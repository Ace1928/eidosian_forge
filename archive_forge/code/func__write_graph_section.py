import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _write_graph_section(self, graph_order):
    """Writes the graph section of the report."""
    self._write_report('%s %s\n' % (_MARKER_SECTION_BEGIN, _SECTION_NAME_GRAPH))
    self._write_report('%s %s\n' % (_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED, not graph_order.contains_cycle))
    l = list(graph_order.topological_order_or_cycle)
    for i in range(0, len(l)):
        self._write_report('%d "%s"\n' % (i, l[i].name))
    self._write_report('%s %s\n' % (_MARKER_SECTION_END, _SECTION_NAME_GRAPH))