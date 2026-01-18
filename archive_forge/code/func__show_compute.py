import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _show_compute(self, show_dataflow):
    """Visualize the computation activity."""
    for dev_stats in self._step_stats.dev_stats:
        device_name = dev_stats.device
        device_pid = self._device_pids[device_name]
        is_gputrace = self._is_gputrace_device(device_name)
        for node_stats in dev_stats.node_stats:
            tid = node_stats.thread_id
            start_time = node_stats.all_start_micros
            end_time = node_stats.all_start_micros + node_stats.all_end_rel_micros
            self._emit_op(node_stats, device_pid, is_gputrace)
            if is_gputrace or node_stats.node_name == 'RecvTensor':
                continue
            _, _, inputs = self._parse_op_label(node_stats.timeline_label)
            for input_name in inputs:
                if input_name not in self._tensors:
                    index = input_name.rfind('/_')
                    if index > 0:
                        input_name = input_name[:index]
                if input_name in self._tensors:
                    tensor = self._tensors[input_name]
                    tensor.add_ref(start_time)
                    tensor.add_unref(end_time - 1)
                    if show_dataflow:
                        create_time, create_pid, create_tid = self._flow_starts[input_name]
                        if create_pid != device_pid or create_tid != tid:
                            flow_id = self._alloc_flow_id()
                            self._chrome_trace.emit_flow_start(input_name, create_time, create_pid, create_tid, flow_id)
                            self._chrome_trace.emit_flow_end(input_name, start_time, device_pid, tid, flow_id)
                else:
                    logging.vlog(1, "Can't find tensor %s - removed by CSE?", input_name)