from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def _populate_recipient_maps(self):
    """Populate the map from node name to recipient(s) of its output(s).

    This method also populates the input map based on reversed ref edges.
    """
    for node in self._node_inputs:
        inputs = self._node_inputs[node]
        for inp in inputs:
            inp = get_node_name(inp)
            if inp not in self._node_recipients:
                self._node_recipients[inp] = []
            self._node_recipients[inp].append(node)
            if inp in self._ref_args:
                if inp not in self._node_reversed_ref_inputs:
                    self._node_reversed_ref_inputs[inp] = []
                self._node_reversed_ref_inputs[inp].append(node)
    for node in self._node_ctrl_inputs:
        ctrl_inputs = self._node_ctrl_inputs[node]
        for ctrl_inp in ctrl_inputs:
            if ctrl_inp in self._copy_send_nodes:
                continue
            if ctrl_inp not in self._node_ctrl_recipients:
                self._node_ctrl_recipients[ctrl_inp] = []
            self._node_ctrl_recipients[ctrl_inp].append(node)