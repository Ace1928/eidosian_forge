import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _validate_dump_with_graphs(self, device_name):
    """Validate the dumped tensor data against the partition graphs.

    Only the watched nodes are validated by this method, because tfdbg allows
    clients to watch only a subset of the nodes.

    Args:
      device_name: (`str`) device name.

    Raises:
      LookupError: If the partition graphs have not been loaded yet.
      ValueError: If dumps contain node names not found in partition graph.
        Or if the temporal order of the dump's timestamps violate the
        input relations on the partition graphs.
    """
    if not self._debug_graphs:
        raise LookupError('No partition graphs loaded for device %s' % device_name)
    debug_graph = self._debug_graphs[device_name]
    for datum in self._dump_tensor_data[device_name]:
        if datum.node_name not in debug_graph.node_inputs:
            raise ValueError("Node name '%s' is not found in partition graphs of device %s." % (datum.node_name, device_name))
    pending_inputs = {}
    for node in debug_graph.node_inputs:
        pending_inputs[node] = []
        inputs = debug_graph.node_inputs[node]
        for inp in inputs:
            inp_node = debug_graphs.get_node_name(inp)
            inp_output_slot = debug_graphs.get_output_slot(inp)
            if inp_node in self._debug_watches[device_name] and inp_output_slot in self._debug_watches[device_name][inp_node] and (debug_graph.node_op_types.get(inp) not in ('Enter', 'NextIteration')) and ((inp_node, inp_output_slot) not in pending_inputs[node]):
                pending_inputs[node].append((inp_node, inp_output_slot))
    for i, datum in enumerate(self._dump_tensor_data[device_name]):
        node = datum.node_name
        slot = datum.output_slot
        if not self._satisfied_at_timestamp(device_name, pending_inputs[node], datum.timestamp, start_i=i + 1):
            raise ValueError('Causality violated in timing relations of debug dumps: %s (%d): these input(s) are not satisfied: %s' % (node, datum.timestamp, repr(pending_inputs[node])))
        recipients = debug_graph.node_recipients[node]
        for recipient in recipients:
            recipient_pending_inputs = pending_inputs[recipient]
            if (node, slot) in recipient_pending_inputs:
                if self.node_op_type(recipient) == 'Merge':
                    del recipient_pending_inputs[:]
                else:
                    del recipient_pending_inputs[recipient_pending_inputs.index((node, slot))]