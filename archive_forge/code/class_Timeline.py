import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
class Timeline(object):
    """A class for visualizing execution timelines of TensorFlow steps."""

    def __init__(self, step_stats, graph=None):
        """Constructs a new Timeline.

    A 'Timeline' is used for visualizing the execution of a TensorFlow
    computation.  It shows the timings and concurrency of execution at
    the granularity of TensorFlow Ops.
    This class is not thread safe.

    Args:
      step_stats: The 'StepStats' proto recording execution times.
      graph: (Optional) The 'Graph' that was executed.
    """
        self._origin_step_stats = step_stats
        self._step_stats = None
        self._graph = graph
        self._chrome_trace = _ChromeTraceFormatter()
        self._next_pid = 0
        self._device_pids = {}
        self._tensor_pids = {}
        self._tensors = {}
        self._next_flow_id = 0
        self._flow_starts = {}
        self._alloc_times = {}
        self._allocator_maximums = {}

    def _alloc_pid(self):
        """Allocate a process Id."""
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def _alloc_flow_id(self):
        """Allocate a flow Id."""
        flow_id = self._next_flow_id
        self._next_flow_id += 1
        return flow_id

    def _parse_op_label(self, label):
        """Parses the fields in a node timeline label."""
        match = re.match('(.*) = (.*)\\((.*)\\)', label)
        if match is None:
            return ('unknown', 'unknown', [])
        nn, op, inputs = match.groups()
        if not inputs:
            inputs = []
        else:
            inputs = inputs.split(', ')
        return (nn, op, inputs)

    def _parse_kernel_label(self, label, node_name):
        """Parses the fields in a node timeline label."""
        start = label.find('@@')
        end = label.find('#')
        if start >= 0 and end >= 0 and (start + 2 < end):
            node_name = label[start + 2:end]
        fields = node_name.split(':') + ['unknown']
        name, op = fields[:2]
        return (name, op)

    def _assign_lanes(self):
        """Assigns non-overlapping lanes for the activities on each device."""
        for device_stats in self._step_stats.dev_stats:
            lanes = [0]
            for ns in device_stats.node_stats:
                l = -1
                for i, lts in enumerate(lanes):
                    if ns.all_start_micros > lts:
                        l = i
                        lanes[l] = ns.all_start_micros + ns.all_end_rel_micros
                        break
                if l < 0:
                    l = len(lanes)
                    lanes.append(ns.all_start_micros + ns.all_end_rel_micros)
                ns.thread_id = l

    def _emit_op(self, nodestats, pid, is_gputrace):
        """Generates a Chrome Trace event to show Op execution.

    Args:
      nodestats: The 'NodeExecStats' proto recording op execution.
      pid: The pid assigned for the device where this op ran.
      is_gputrace: If True then this op came from the GPUTracer.
    """
        node_name = nodestats.node_name
        start = nodestats.all_start_micros
        duration = nodestats.all_end_rel_micros
        tid = nodestats.thread_id
        inputs = []
        if is_gputrace:
            node_name, op = self._parse_kernel_label(nodestats.timeline_label, node_name)
        elif node_name == 'RecvTensor':
            op = 'RecvTensor'
        else:
            _, op, inputs = self._parse_op_label(nodestats.timeline_label)
        args = {'name': node_name, 'op': op}
        if build_info.build_info['is_rocm_build']:
            args['kernel'] = nodestats.timeline_label.split('@@')[0]
        for i, iname in enumerate(inputs):
            args['input%d' % i] = iname
        self._chrome_trace.emit_region(start, duration, pid, tid, 'Op', op, args)

    def _emit_tensor_snapshot(self, tensor, timestamp, pid, tid, value):
        """Generate Chrome Trace snapshot event for a computed Tensor.

    Args:
      tensor: A 'TensorTracker' object.
      timestamp:  The timestamp of this snapshot as a long integer.
      pid: The pid assigned for showing the device where this op ran.
      tid: The tid of the thread computing the tensor snapshot.
      value: A JSON-compliant snapshot of the object.
    """
        desc = str(value.tensor_description).replace('"', '')
        snapshot = {'tensor_description': desc}
        self._chrome_trace.emit_obj_snapshot('Tensor', tensor.name, timestamp, pid, tid, tensor.object_id, snapshot)

    def _produce_tensor(self, name, timestamp, tensors_pid, allocator, num_bytes):
        object_id = len(self._tensors)
        tensor = _TensorTracker(name, object_id, timestamp, tensors_pid, allocator, num_bytes)
        self._tensors[name] = tensor
        return tensor

    def _is_gputrace_device(self, device_name):
        """Returns true if this device is part of the GPUTracer logging."""
        return '/stream:' in device_name or '/memcpy' in device_name

    def _allocate_pids(self):
        """Allocate fake process ids for each device in the StepStats."""
        self._allocators_pid = self._alloc_pid()
        self._chrome_trace.emit_pid('Allocators', self._allocators_pid)
        for dev_stats in self._step_stats.dev_stats:
            device_pid = self._alloc_pid()
            self._device_pids[dev_stats.device] = device_pid
            tensors_pid = self._alloc_pid()
            self._tensor_pids[dev_stats.device] = tensors_pid
            self._chrome_trace.emit_pid(dev_stats.device + ' Compute', device_pid)
            self._chrome_trace.emit_pid(dev_stats.device + ' Tensors', tensors_pid)

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

    def _show_memory_counters(self):
        """Produce a counter series for each memory allocator."""
        allocations = {}
        for name in self._tensors:
            tensor = self._tensors[name]
            self._chrome_trace.emit_obj_delete('Tensor', name, tensor.last_unref, tensor.pid, 0, tensor.object_id)
            allocator = tensor.allocator
            if allocator not in allocations:
                allocations[allocator] = []
            num_bytes = tensor.num_bytes
            allocations[allocator].append((tensor.create_time, num_bytes, name))
            allocations[allocator].append((tensor.last_unref, -num_bytes, name))
        alloc_maxes = {}
        for allocator in allocations:
            alloc_list = allocations[allocator]
            alloc_list.sort()
            total_bytes = 0
            alloc_tensor_set = set()
            alloc_maxes[allocator] = AllocationMaximum(timestamp=0, num_bytes=0, tensors=set())
            for time, num_bytes, name in sorted(alloc_list, key=lambda allocation: allocation[0]):
                total_bytes += num_bytes
                if num_bytes < 0:
                    alloc_tensor_set.discard(name)
                else:
                    alloc_tensor_set.add(name)
                if total_bytes > alloc_maxes[allocator].num_bytes:
                    alloc_maxes[allocator] = AllocationMaximum(timestamp=time, num_bytes=total_bytes, tensors=copy.deepcopy(alloc_tensor_set))
                self._chrome_trace.emit_counter('Memory', allocator, self._allocators_pid, time, allocator, total_bytes)
        self._allocator_maximums = alloc_maxes

    def _preprocess_op_time(self, op_time):
        """Update the start and end time of ops in step stats.

    Args:
    op_time: How the execution time of op is shown in timeline. Possible values
      are "schedule", "gpu" and "all". "schedule" will show op from the time it
      is scheduled to the end of the scheduling. Notice by the end of its
      scheduling its async kernels may not start yet. It is shown using the
      default value from step_stats. "gpu" will show op with the execution time
      of its kernels on GPU. "all" will show op from the start of its scheduling
      to the end of its last kernel.
    """
        if op_time == 'schedule':
            self._step_stats = self._origin_step_stats
            return
        self._step_stats = copy.deepcopy(self._origin_step_stats)
        stream_all_stats = []
        job_stats = []
        for stats in self._step_stats.dev_stats:
            if '/stream:all' in stats.device:
                stream_all_stats.append(stats)
            elif '/job' in stats.device:
                job_stats.append(stats)
        op_gpu_start = {}
        op_gpu_end = {}
        for stats in stream_all_stats:
            for kernel in stats.node_stats:
                name, _ = self._parse_kernel_label(kernel.timeline_label, kernel.node_name)
                start = kernel.all_start_micros
                end = kernel.all_start_micros + kernel.all_end_rel_micros
                if name in op_gpu_start:
                    op_gpu_start[name] = min(op_gpu_start[name], start)
                    op_gpu_end[name] = max(op_gpu_end[name], end)
                else:
                    op_gpu_start[name] = start
                    op_gpu_end[name] = end
        for stats in job_stats:
            for op in stats.node_stats:
                if op.node_name in op_gpu_start:
                    end = max(op_gpu_end[op.node_name], op.all_start_micros + op.all_end_rel_micros)
                    if op_time == 'gpu':
                        op.all_start_micros = op_gpu_start[op.node_name]
                    op.all_end_rel_micros = end - op.all_start_micros

    def analyze_step_stats(self, show_dataflow=True, show_memory=True, op_time='schedule'):
        """Analyze the step stats and format it into Chrome Trace Format.

    Args:
      show_dataflow: (Optional.) If True, add flow events to the trace
        connecting producers and consumers of tensors.
      show_memory: (Optional.) If True, add object snapshot events to the trace
        showing the sizes and lifetimes of tensors.
      op_time: (Optional.) How the execution time of op is shown in timeline.
        Possible values are "schedule", "gpu" and "all". "schedule" will show op
        from the time it is scheduled to the end of the scheduling. Notice by
        the end of its scheduling its async kernels may not start yet. It is
        shown using the default value from step_stats. "gpu" will show op with
        the execution time of its kernels on GPU. "all" will show op from the
        start of its scheduling to the end of its last kernel.

    Returns:
      A 'StepStatsAnalysis' object.
    """
        self._preprocess_op_time(op_time)
        self._allocate_pids()
        self._assign_lanes()
        self._analyze_tensors(show_memory)
        self._show_compute(show_dataflow)
        if show_memory:
            self._show_memory_counters()
        return StepStatsAnalysis(chrome_trace=self._chrome_trace, allocator_maximums=self._allocator_maximums)

    def generate_chrome_trace_format(self, show_dataflow=True, show_memory=False, op_time='schedule'):
        """Produces a trace in Chrome Trace Format.

    Args:
      show_dataflow: (Optional.) If True, add flow events to the trace
        connecting producers and consumers of tensors.
      show_memory: (Optional.) If True, add object snapshot events to the trace
        showing the sizes and lifetimes of tensors.
      op_time: (Optional.) How the execution time of op is shown in timeline.
        Possible values are "schedule", "gpu" and "all".
        "schedule" will show op from the time it is scheduled to the end of
          the scheduling.
          Notice by the end of its scheduling its async kernels may not start
          yet. It is shown using the default value from step_stats.
        "gpu" will show op with the execution time of its kernels on GPU.
        "all" will show op from the start of its scheduling to the end of
          its last kernel.

    Returns:
      A JSON formatted string in Chrome Trace format.
    """
        step_stats_analysis = self.analyze_step_stats(show_dataflow=show_dataflow, show_memory=show_memory, op_time=op_time)
        return step_stats_analysis.chrome_trace.format_to_string(pretty=True)