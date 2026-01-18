import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _analyze_tensors(self, show_memory):
    """Analyze tensor references to track dataflow."""
    for dev_stats in self._step_stats.dev_stats:
        device_pid = self._device_pids[dev_stats.device]
        tensors_pid = self._tensor_pids[dev_stats.device]
        for node_stats in dev_stats.node_stats:
            tid = node_stats.thread_id
            node_name = node_stats.node_name
            start_time = node_stats.all_start_micros
            end_time = node_stats.all_start_micros + node_stats.all_end_rel_micros
            for index, output in enumerate(node_stats.output):
                if index:
                    output_name = '%s:%d' % (node_name, index)
                else:
                    output_name = node_name
                allocation = output.tensor_description.allocation_description
                num_bytes = allocation.requested_bytes
                allocator_name = allocation.allocator_name
                tensor = self._produce_tensor(output_name, start_time, tensors_pid, allocator_name, num_bytes)
                tensor.add_ref(start_time)
                tensor.add_unref(end_time)
                self._flow_starts[output_name] = (end_time, device_pid, tid)
                if show_memory:
                    self._chrome_trace.emit_obj_create('Tensor', output_name, start_time, tensors_pid, tid, tensor.object_id)
                    self._emit_tensor_snapshot(tensor, end_time - 1, tensors_pid, tid, output)