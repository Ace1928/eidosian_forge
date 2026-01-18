from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def _process_debug_graph_node(self, node):
    """Process a node from the debug GraphDef.

    Args:
      node: (NodeDef) A partition-graph node to be processed.

    Raises:
      ValueError: If duplicate node names are encountered.
    """
    if is_debug_node(node.name):
        return
    if node.name in self._node_inputs:
        raise ValueError("Duplicate node name on device %s: '%s'" % (self._device_name, node.name))
    self._node_attributes[node.name] = node.attr
    self._node_inputs[node.name] = []
    self._node_ctrl_inputs[node.name] = []
    self._node_recipients[node.name] = []
    self._node_ctrl_recipients[node.name] = []
    if node.name not in self._node_devices:
        self._node_devices[node.name] = set()
    self._node_devices[node.name].add(node.device if node.device else self._device_name)
    self._node_op_types[node.name] = node.op
    self._ref_args[node.name] = self._get_ref_args(node)
    for inp in node.input:
        if is_copy_node(inp) and (node.op == '_Send' or node.op == '_Retval'):
            self._copy_send_nodes.append(node.name)
        if inp.startswith('^'):
            cinp = inp[1:]
            self._node_ctrl_inputs[node.name].append(cinp)
        else:
            self._node_inputs[node.name].append(inp)