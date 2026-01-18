from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
class DFSGraphTracer(object):
    """Graph input tracer using depth-first search."""

    def __init__(self, input_lists, skip_node_names=None, destination_node_name=None):
        """Constructor of _DFSGraphTracer.

    Args:
      input_lists: A list of dicts. Each dict is an adjacency (input) map from
        the recipient node name as the key and the list of input node names
        as the value.
      skip_node_names: Optional: a list of node names to skip tracing.
      destination_node_name: Optional: destination node name. If not `None`, it
        should be the name of a destination not as a str and the graph tracing
        will raise GraphTracingReachedDestination as soon as the node has been
        reached.

    Raises:
      GraphTracingReachedDestination: if stop_at_node_name is not None and
        the specified node is reached.
    """
        self._input_lists = input_lists
        self._skip_node_names = skip_node_names
        self._inputs = []
        self._visited_nodes = []
        self._depth_count = 0
        self._depth_list = []
        self._destination_node_name = destination_node_name

    def trace(self, graph_element_name):
        """Trace inputs.

    Args:
      graph_element_name: Name of the node or an output tensor of the node, as a
        str.

    Raises:
      GraphTracingReachedDestination: if destination_node_name of this tracer
        object is not None and the specified node is reached.
    """
        self._depth_count += 1
        node_name = get_node_name(graph_element_name)
        if node_name == self._destination_node_name:
            raise GraphTracingReachedDestination()
        if node_name in self._skip_node_names:
            return
        if node_name in self._visited_nodes:
            return
        self._visited_nodes.append(node_name)
        for input_list in self._input_lists:
            if node_name not in input_list:
                continue
            for inp in input_list[node_name]:
                if get_node_name(inp) in self._visited_nodes:
                    continue
                self._inputs.append(inp)
                self._depth_list.append(self._depth_count)
                self.trace(inp)
        self._depth_count -= 1

    def inputs(self):
        return self._inputs

    def depth_list(self):
        return self._depth_list