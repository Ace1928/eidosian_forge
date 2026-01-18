import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def _trace_execution(self, graph, tensor_fetches, op_fetches=None, on_tpu=True):
    """Commong tracing function for both CPU and TPUs.

    The caller function should set device_type, num_replicas,
    num_replicas_per_host, num_hosts and replica_id before calling
    _trace_execution.


    Args:
      graph: the graph of Ops executed on the TPU.
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.
      on_tpu: True if executing on TPU.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    Raises:
      RuntimeError: If tensor_fetches is None or empty.
    """

    def _cast_unsupported_dtypes(tensor):
        """Casts tensor to a supported type."""
        if tensor.dtype.__eq__(dtypes.int64):
            return math_ops.cast(tensor, dtypes.int32)
        if tensor.dtype.__eq__(dtypes.bfloat16) or tensor.dtype.__eq__(dtypes.float16):
            return math_ops.cast(tensor, dtypes.float32)
        return tensor
    trace_mode = self._parameters.trace_mode
    device_type = self._tt_config.device_type
    self._outmost_context = graph._get_control_flow_context()
    analytics.track_usage('tensor_tracer', [trace_mode, device_type])
    TensorTracer.check_device_type(device_type)
    TensorTracer.check_trace_mode(device_type, trace_mode)
    processed_t_fetches = self._process_tensor_fetches(tensor_fetches)
    op_fetches = self._process_op_fetches(op_fetches)
    all_fetches = op_fetches + [tensor.op for tensor in processed_t_fetches]
    exec_op_set = self._filter_execution_path_operations(graph.get_operations(), all_fetches)
    graph_summary_tag = _graph_summary_tag(graph)
    tensor_trace_order = self._determine_trace_and_create_report(graph, exec_op_set, graph_summary_tag)
    tensor_fetch_set = set(processed_t_fetches)
    tracing_ops = []
    sorted_exec_op_list = list(exec_op_set)
    sorted_exec_op_list.sort(key=lambda op: op.name)
    for op in sorted_exec_op_list:
        for i in range(len(op.outputs)):
            out_tensor = op.outputs[i]
            tensor_name = out_tensor.name
            if tensor_name not in tensor_trace_order.tensorname_to_cache_idx:
                continue
            self._traced_op_names.add(op.name)
            consumers = out_tensor.consumers()
            consumers = [cop for cop in consumers if cop in exec_op_set]
            is_a_fetched_tensor = out_tensor in tensor_fetch_set
            if not consumers and (not is_a_fetched_tensor):
                continue
            op_control_flow_context = self._get_op_control_flow_context(op)
            if op_control_flow_context:
                graph._set_control_flow_context(op_control_flow_context)
            processed_tensors = self._preprocess_traced_tensor(out_tensor)
            if on_tpu:
                for signature in processed_tensors.keys():
                    processed_tensors[signature] = _cast_unsupported_dtypes(processed_tensors[signature])
            if self._use_tensor_values_cache():
                if self._use_temp_cache():
                    cache_idx = tensor_trace_order.tensorname_to_cache_idx[tensor_name]
                    self._save_tensor_value_to_tmp_cache(cache_idx, processed_tensors, graph)
                    trace_op = None
                else:
                    cache_idx = tensor_trace_order.tensorname_to_cache_idx[tensor_name]
                    trace_op = self._save_tensor_value_to_cache_op(cache_idx, processed_tensors, graph)
            elif self._use_tensor_buffer():
                if len(processed_tensors) != 1:
                    raise RuntimeError('Multiple stats are only allowed in compact mode.')
                processed_out_tensor = list(processed_tensors.values())[0]
                trace_op = self._snapshot_tensor(processed_out_tensor)
            else:

                def tpu_wrap_trace_fn(tensor, out_tensor_name):
                    """Wraps the trace_fn with outside compilation if on TPUs."""
                    tensor_trace_fn = self._make_tensor_trace_fun(out_tensor_name, tensor_trace_order)
                    if on_tpu:
                        return tpu_replication.outside_compilation(tensor_trace_fn, tensor)
                    else:
                        return tensor_trace_fn(tensor)
                if len(processed_tensors) != 1:
                    raise RuntimeError('Multiple stats are only allowed in compact mode.')
                processed_out_tensor = next(iter(processed_tensors.values()))
                trace_op = tpu_wrap_trace_fn(processed_out_tensor, tensor_name)
            if op_control_flow_context:
                graph._set_control_flow_context(self._outmost_context)
            if trace_op:
                if is_a_fetched_tensor:
                    tracing_ops.append(trace_op)
                    continue
                for consumer_op in consumers:
                    consumer_op._add_control_input(trace_op)
    graph._set_control_flow_context(self._outmost_context)
    if tracing_ops:
        processed_t_fetches = control_flow_ops.tuple(processed_t_fetches, control_inputs=tracing_ops)
    if self._use_tensor_values_cache() or self._use_tensor_buffer():
        if self._use_temp_cache():
            graph_cache_var = self._cache_variable_for_graph(graph)
            if graph not in self._temp_cache_var:
                raise RuntimeError('graph is not in self._temp_cache_var')
            graph_cache_var[_TT_SUMMARY_TAG] = array_ops_stack.stack(self._temp_cache_var[graph], axis=0, name='stack_all_op_signatures')
        if self._create_host_call():
            self._prepare_host_call_fn(processed_t_fetches, op_fetches, graph, graph_summary_tag)
            if not on_tpu:
                write_cache, caches_to_write = self._host_call_fn[_TT_HOSTCALL_KEY]
                cache_write_op = write_cache(**caches_to_write)
                processed_t_fetches = control_flow_ops.tuple(processed_t_fetches, control_inputs=[cache_write_op])
                del self._host_call_fn[_TT_HOSTCALL_KEY]
            elif self._parameters.flush_summaries_with_outside_compile:
                write_cache, caches_to_write = self._host_call_fn[_TT_HOSTCALL_KEY]
                if _TT_SUMMARY_TAG in caches_to_write and 'step' in caches_to_write:
                    step = caches_to_write['step']
                    tensor_tracer_summary = caches_to_write[_TT_SUMMARY_TAG]
                    tt_core_summary = self.merge_caches_on_tpu(tensor_tracer_summary[0])
                    if not self._parameters.collect_summary_per_core:
                        tt_core_summary = self.aggregate_global_cache(tt_core_summary)

                    def write_if_core_0(step, replica_id, tt_summary):
                        return cond.cond(math_ops.equal(replica_id, 0), lambda: write_cache(step=step, event_file_suffix=None, tensor_tracer_summary=tt_summary), control_flow_ops.no_op)
                    write_op = tpu_replication.outside_compilation(write_if_core_0, step=step, replica_id=self._replica_id, tt_summary=tt_core_summary)
                    processed_t_fetches = control_flow_ops.tuple(processed_t_fetches, control_inputs=[write_op])
                    del self._host_call_fn[_TT_HOSTCALL_KEY]
                else:
                    raise ValueError('Outside compiled flush in only supported for summary mode')
        else:
            processed_t_fetches = self._flush_tensor_values_cache(processed_t_fetches, op_fetches, on_tpu=on_tpu, tensor_trace_order=tensor_trace_order, graph=graph)
    return self._convert_fetches_to_input_format(tensor_fetches, processed_t_fetches)