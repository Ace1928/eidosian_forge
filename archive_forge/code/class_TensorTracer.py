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
class TensorTracer:
    """A software construct for tracing tensor values in a TF graph.

  This utility is disabled by default. It is hooked into tpu.rewrite, so it can
  easily be enabled on TPUs by setting the TENSOR_TRACER_FLAGS env variable as
  below without a code change.
    export TENSOR_TRACER_FLAGS="--enable=1"

  Below is the use example to enable it on CPUs or GPUs, or for more advance use
  cases on TPUs.

    a = x + 1
    b = a * 2
    rs = tf.reduce_sum(b)
    tensor_tracer.set_parameters({'trace_dir': 'path/to/trace_dir',
                             'report_file: 'path/to/report/file'})
    tt = tensor_tracer.TensorTracer()
    if on_tpu:
      rs = tt.trace_tpu(tf.get_default_graph(),
                          tensor_fetches=rs)
    else:
      rs = tt.trace_cpu(tf.get_default_graph(),
                          tensor_fetches=rs)
    session.run(rs)

  If it is enabled, it will trace the output tensor values of
  selected Ops in the graph. It has two outputs: (1) the traces and (2)
  a report. The traces are dumped to a specified directory during the graph
  execution, while the report is dumped during the graph construction.
  By passing options via the env variable, users can change:
     (1) the trace mode (e.g., detecting NaN/Inf, printing partial or
         full tensor values)
     (2) which Ops to be traced (via op.name or op.type)
     (3) output trace file path.

  """
    _traced_graphs = set()

    @staticmethod
    def is_enabled():
        """Returns True if TensorTracer is enabled."""
        try:
            enable = tensor_tracer_flags.TTParameters().is_enabled()
            if enable:
                tt_gauge.get_cell('is_enabled').set(True)
            return enable
        except (ValueError, RuntimeError) as e:
            logging.warning('Tensor Tracer V1 flags processing error encountered in is_enabled check. %s', e)
            return True

    @staticmethod
    def check_device_type(device_type):
        """Checks if the given device type is valid."""
        if device_type not in (_DEVICE_TYPE_TPU, _DEVICE_TYPE_CPU):
            raise ValueError('Invalid device_type "%s"' % device_type)

    @staticmethod
    def check_trace_mode(device_type, trace_mode):
        """Checks if the given trace mode work on the given device type.

    Args:
      device_type: Device type, TPU, GPU, CPU.
      trace_mode: Tensor tracer trace mode.
    Raises:
      ValueError: If the given trace mode is not supported for the device.
    """
        if trace_mode == tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY:
            if device_type != _DEVICE_TYPE_TPU:
                raise ValueError('Device_type "%s" is not yet supported for trace mode "%s"' % (device_type, trace_mode))

    @staticmethod
    def loop_cond_op(op):
        return op.type in ('LoopCond', 'RefLoopCond')

    @staticmethod
    def while_loop_op(op):
        """Returns true if op is one of the special ops of in a while loop.

    Args:
       op: A tf.Operation.

    Returns:
       True if the given op is one of [Switch, Merge, Enter, Exit,
       NextIteration, LoopCond], which are all building blocks for TF while
       loops.
    """
        return control_flow_util.IsLoopSwitch(op) or control_flow_util.IsLoopMerge(op) or control_flow_util.IsLoopEnter(op) or control_flow_util.IsLoopExit(op) or TensorTracer.loop_cond_op(op) or (op.type in ('RefNextIteration', 'NextIteration'))

    @staticmethod
    def control_flow_op(op):
        """Returns true if op is one of the special ops of in a while loop.

    Args:
       op: A tf.Operation.

    Returns:
       True if the given op is one of [Switch, Merge, Enter, Exit,
       NextIteration, LoopCond], which are all building blocks for TF while
       loops.
    """
        return control_flow_util.IsSwitch(op) or control_flow_util.IsMerge(op)

    @staticmethod
    def unsafe_op(op):
        """Returns True if this op is not safe to be traced."""
        if op.type == 'Assign':
            return True
        return False

    @staticmethod
    def device_mismatch(device_type, op):
        if device_type == _DEVICE_TYPE_TPU:
            return tpu_replication._TPU_REPLICATE_ATTR not in op.node_def.attr
        return False

    @staticmethod
    def unsafe_scalar_trace(op):
        """Return true if scalar output tensor from Op is not safe to be traced."""
        if op.type in ('LoopCond', 'Enter', 'Merge', 'Const', 'Switch', 'Less', 'ReadVariableOp'):
            return True
        if op.type in ('VarHandleOp', 'IteratorToStringHandle', 'IteratorGetNext', 'OneShotIterator', 'IteratorV2', 'MakeIterator', 'BatchDatasetV2', 'MapDataset', 'FixedLengthRecordDataset', 'TakeDataset', 'ZipDataset', 'Placeholder', 'PlaceholderWithDefault', 'StridedSlice'):
            return True
        return False

    def _is_interesting_op(self, op):
        """Returns True if the given op is not an interesting one to be traced."""
        return op_priority(op.type) <= self._parameters.trace_level

    @staticmethod
    def reason(op_idx, details):
        """Returns reason why the Op at op_idx is traced or not."""
        return '%d %s' % (op_idx, details)

    def __init__(self):
        """Initializes a TensorTracer.

    Sets the various member fields from the flags (if given) or the defaults.
    """
        self._replica_id = None
        self._tt_config = tensor_tracer_report.TensorTracerConfig()
        self._parameters = tensor_tracer_flags.TTParameters()
        self._host_call_fn = {}
        self._cache_variables = {}
        self._history_value_cache = {}
        self._traced_op_names = set()
        self._report_proto = None
        self._temp_cache_var = {}
        self._report_proto_path = ''
        self._outmost_context = None

    def report_proto(self):
        """Getter for tensor_tracer.proto object for summary and full_tensor_summary modes.

    Returns:
      A tensor_tracer.proto object.
    Raises:
      ValueError if called before tracing happens, or when trace mode is not
      summary or full_tensor_summary.
    """
        if self._report_proto:
            return self._report_proto
        else:
            raise ValueError('Call to report_proto must be done after tracing.Report proto only exists for trace_mode=[summary|full_tensor_summary]')

    def report_proto_path(self):
        """Getter for path where tensor_tracer.proto object should be written.

    Returns:
      A string path.
    """
        return self._report_proto_path

    def _escape_namescopes(self, variable_name):
        return variable_name.replace('/', '_').replace(':', '_')

    def _cache_variable_for_graph(self, graph):
        if graph not in self._cache_variables:
            self._cache_variables[graph] = {}
        return self._cache_variables[graph]

    def _create_or_get_tensor_history_values_cache(self, cache_name, graph, shape=None, dtype=dtypes.float32):
        """Creates a variable as the cache to store historic intermediate tensor values.

    Args:
      cache_name: Name to be given to the cache (an instance of tf.variable).
      graph: Tensorflow graph.
      shape: A list of dimensions.
      dtype: Data type of created cache.
    Returns:
      A ref to newly created or existing cache with the given dimensions.
    Raises:
      ValueError:
        (1) If graph is None, or
        (2) shape is None when a new cache needs to be created.
    """
        if graph is None:
            raise ValueError('Invalid graph.')
        if graph not in self._history_value_cache:
            self._history_value_cache[graph] = {}
        if cache_name not in self._history_value_cache[graph]:
            if shape is None:
                raise ValueError('shape must be provided at cache creation.')
            if dtype.is_integer:
                init_val = int(_COMPACT_TRACE_ENTRY_INIT_VALUE)
            else:
                init_val = _COMPACT_TRACE_ENTRY_INIT_VALUE
            with graph.as_default() as g, g.name_scope(None):
                self._history_value_cache[graph][cache_name] = variable_scope.get_variable('tt_history' + '_' + self._escape_namescopes(cache_name), shape=shape, dtype=dtype, initializer=init_ops.constant_initializer(init_val), trainable=False, use_resource=True, collections=[_TENSOR_TRACER_STORAGE, ops.GraphKeys.LOCAL_VARIABLES])
        return self._history_value_cache[graph][cache_name]

    def _create_or_get_tensor_values_cache(self, cache_name, graph, shape=None, dtype=dtypes.float32):
        """Creates a variable as the cache to store intermediate tensor values.

    Args:
      cache_name: Name to be given to the cache (an instance of tf.variable).
      graph: Tensorflow graph.
      shape: A list of dimensions.
      dtype: Data type of created cache.
    Returns:
      A ref to newly created or existing cache with the given dimensions.
    Raises:
      ValueError:
        (1) If graph is None, or
        (2) shape is None when a new cache needs to be created.
    """
        if graph is None:
            raise ValueError('Invalid graph.')
        graph_cache_var = self._cache_variable_for_graph(graph)
        if cache_name not in graph_cache_var:
            if shape is None:
                raise ValueError('shape must be provided at cache creation.')
            if dtype.is_integer:
                init_val = int(_COMPACT_TRACE_ENTRY_INIT_VALUE)
            else:
                init_val = _COMPACT_TRACE_ENTRY_INIT_VALUE
            with graph.as_default() as g, g.name_scope(None):
                graph_cache_var[cache_name] = variable_scope.get_variable(_TT_SNAPSHOT + '_' + self._escape_namescopes(cache_name), shape=shape, dtype=dtype, initializer=init_ops.constant_initializer(init_val), trainable=False, use_resource=True, collections=[_TENSOR_TRACER_STORAGE, ops.GraphKeys.LOCAL_VARIABLES])
        return graph_cache_var[cache_name]

    def _add_replica_id_to_graph(self):
        """Adds nodes for computing the replica ID to the graph."""
        if self._tt_config.num_replicas:
            with ops.control_dependencies(None):
                self._replica_id = tpu_ops.tpu_replicated_input(list(range(self._tt_config.num_replicas)), name='tt_replica_id')
        else:
            self._replica_id = 'unknown'

    def _inside_op_range(self, idx):
        """Return True if the given index is inside the selected range."""
        if idx < self._parameters.op_range[0]:
            return False
        return self._parameters.op_range[1] < 0 or idx <= self._parameters.op_range[1]

    def _is_user_included_op(self, op):
        """Checks whether the op is included in the tensor tracer flags.

    Args:
      op: tf Operation
    Returns:
      True, if the op is included.
      An op is included if:
      - Its op name is given in included_opnames
      - Its op type is given in included_optypes
      - The op is at most _trace_ops_before_included hops before an included op
      - The op is at most _trace_ops_after_included hops after an included op
    """
        for opname_re in self._parameters.included_opname_re_list:
            if opname_re.match(op.name):
                return True
        for optype_re in self._parameters.included_optype_re_list:
            if optype_re.match(op.type):
                return True
        return False

    def _is_user_excluded_op(self, op):
        for opname_re in self._parameters.excluded_opname_re_list:
            if opname_re.match(op.name):
                return True
        for optype_re in self._parameters.excluded_optype_re_list:
            if optype_re.match(op.type):
                return True
        return False

    def _signature_types(self):
        """Returns a dictionary holding the order of signatures in the cache for the selected trace mode."""
        if self._parameters.trace_mode in set([tensor_tracer_flags.TRACE_MODE_NAN_INF, tensor_tracer_flags.TRACE_MODE_NORM, tensor_tracer_flags.TRACE_MODE_HISTORY, tensor_tracer_flags.TRACE_MODE_MAX_ABS]):
            return {self._parameters.trace_mode: 0}
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
            return self._parameters.summary_signatures
        return {}

    def _num_signature_dimensions(self):
        return len(self._signature_types())

    def _use_temp_cache(self):
        """Returns true if the intermediate values should be stacked instead of being stored in a tf.Variable.

    Returns:
      A boolean, denoting whether to use a temporary cache or not.
    """
        if self._use_tensor_buffer():
            return False
        if self._use_tensor_values_cache():
            return self._parameters.use_temp_cache_var
        else:
            return False

    def _use_tensor_values_cache(self):
        """Returns True if immediate tensors should be first saved to a cache."""
        return self._parameters.use_compact_trace

    def _use_tensor_buffer(self):
        """Returns true if the whole tensor needs to be cached/buffered in memory."""
        return self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY

    def _merge_tensor_signatures(self, signatures):
        """Returns a tensor that merges the given signatures.

    Args:
      signatures: A dictionary of the signature updates from signature name to
      a tensor of dimension [1].
    Returns:
      A tensor that concats the signature values in a predefined order.
    Raises:
      ValueError: Unable to merge signatures.
    """
        sorted_update = []
        if self._num_signature_dimensions() > 1:
            signature_indices = self._signature_types()
            for _, val in sorted(signatures.items(), key=lambda item: signature_indices[item[0]]):
                sorted_update.append(val)
            updates = array_ops_stack.stack(sorted_update, axis=0, name='merge_single_op_signatures')
        elif self._num_signature_dimensions() == 1:
            (_, val), = signatures.items()
            updates = val
        else:
            raise ValueError('Cannot merge 0 signatures. Check the value passed for flag --signatures.')
        return updates

    def _save_tensor_value_to_tmp_cache(self, cache_idx, updates, graph):
        """Returns an op that will save the given updates to an entry in the cache.

    Args:
      cache_idx: The cache index of the tensor within the cache.
      updates: A dictionary of the signature updates from signature name to
      a tensor of dimension [1].
      graph: A TensorFlow graph.
    Raises:
      RuntimeError:
        (1) graph is not already in self._temp_cache_var, or
        (2) cache_idx is out of range.
    """
        updates = self._merge_tensor_signatures(updates)
        updates = array_ops.reshape(updates, [self._num_signature_dimensions()])
        if graph not in self._temp_cache_var:
            raise RuntimeError('graph is not in self._temp_cache_var')
        if cache_idx >= len(self._temp_cache_var[graph]):
            raise RuntimeError('cache_idx (%d) is out of range (%d)' % (cache_idx, len(self._temp_cache_var[graph])))
        self._temp_cache_var[graph][cache_idx] = updates

    def _save_tensor_value_to_cache_op(self, cache_idx, updates, graph):
        """Returns an op that will save the given updates to an entry in the cache.

    Args:
      cache_idx: The cache index of the tensor within the cache.
      updates: A dictionary of the signature updates.
      graph: A TensorFlow graph.
    Returns:
      Cache update operation.
    """
        updates = self._merge_tensor_signatures(updates)
        updates = array_ops.reshape(updates, [1, self._num_signature_dimensions()])
        indices = constant_op.constant([cache_idx])
        cache = self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG, graph)
        return state_ops.scatter_update(cache, indices, updates).op

    def _snapshot_tensor(self, tensor):
        """Creates a new tf.Variable and a new tf.Operation that assigns the value of the tensor to this variable.

    Args:
      tensor: tensor whose values will be stored in a new tf.Variable.
    Returns:
      An assignment operation.
    """
        snapshot_variable = self._create_or_get_tensor_values_cache(tensor.name, tensor.op.graph, tensor.shape.as_list(), tensor.dtype)
        return state_ops.assign(snapshot_variable, tensor).op

    def _preprocess_traced_tensor(self, tensor):
        """Computes NAN/Norm/Max on TPUs before sending to CPU.

    Args:
      tensor: The tensor to be traced.
    Returns:
      A tensor that should be input to the trace_function.
    Raises:
      RuntimeError: If the signature is invalid.
    """

        def _detect_nan_inf(tensor):
            """Trace function for detecting any NaN/Inf in the tensor."""
            if tensor.dtype.is_floating:
                mask = math_ops.reduce_any(gen_math_ops.logical_or(gen_math_ops.is_nan(tensor), gen_math_ops.is_inf(tensor)))
                output_tensor = cond.cond(mask, lambda: constant_op.constant([1.0]), lambda: constant_op.constant([0.0]))
            else:
                output_tensor = constant_op.constant([0.0])
            return output_tensor

        def _compute_signature(tensor, tf_op, cast_to_f32=True):
            if cast_to_f32:
                tensor = math_ops.cast(tensor, dtypes.float32)
            output_tensor = tf_op(tensor)
            if not output_tensor.get_shape().is_fully_defined():
                output_tensor = array_ops.reshape(output_tensor, [])
            return output_tensor

        def _show_size(tensor):
            tsize = _compute_signature(tensor, array_ops.size, cast_to_f32=False)
            return math_ops.cast(tsize, dtypes.float32)

        def _show_max(tensor, cast_to_f32=True):
            return _compute_signature(tensor, math_ops.reduce_max, cast_to_f32)

        def _show_min(tensor, cast_to_f32=True):
            return _compute_signature(tensor, math_ops.reduce_min, cast_to_f32)

        def _show_norm(tensor, cast_to_f32=True):
            return _compute_signature(tensor, linalg_ops.norm, cast_to_f32)

        def _show_sparsity(tensor, cast_to_f32=True, tolerance=1e-06):

            def sparsity_fn(tensor):
                non_zeros = math_ops.greater_equal(math_ops.abs(tensor), tolerance)
                nans = math_ops.is_nan(tensor)
                return nn_impl.zero_fraction(math_ops.logical_or(non_zeros, nans))
            return _compute_signature(tensor, sparsity_fn, cast_to_f32)

        def _show_mean_and_variance(tensor, cast_to_f32=True):
            """Returns the mean and variance of the given tensor."""
            if cast_to_f32:
                tensor = math_ops.cast(tensor, dtypes.float32)
            mean, var = nn_impl.moments(array_ops.reshape(tensor, [-1]), axes=[0])
            if not mean.get_shape().is_fully_defined():
                mean = array_ops.reshape(mean, [])
            if not var.get_shape().is_fully_defined():
                var = array_ops.reshape(var, [])
            return (mean, var)

        def _show_max_abs(tensor, cast_to_f32=True):
            return _compute_signature(tensor, lambda t: math_ops.reduce_max(math_ops.abs(t)), cast_to_f32)
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
            return {self._parameters.trace_mode: _detect_nan_inf(tensor)}
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_PART_TENSOR:
            return {self._parameters.trace_mode: tensor}
        if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_FULL_TENSOR, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY):
            return {self._parameters.trace_mode: tensor}
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NORM:
            return {self._parameters.trace_mode: array_ops.reshape(_show_norm(tensor), [1])}
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_HISTORY:
            return {self._parameters.trace_mode: array_ops.reshape(_show_norm(tensor), [1])}
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_MAX_ABS:
            return {self._parameters.trace_mode: _show_max_abs(tensor)}
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
            tensor = math_ops.cast(tensor, dtypes.float32)
            result_dict = {}
            if _TT_SUMMARY_MEAN in self._signature_types() or _TT_SUMMARY_VAR in self._signature_types():
                mean, variance = _show_mean_and_variance(tensor, cast_to_f32=False)
            for signature_name, _ in sorted(self._signature_types().items(), key=lambda x: x[1]):
                if signature_name == _TT_SUMMARY_NORM:
                    signature_result_tensor = _show_norm(tensor, cast_to_f32=False)
                elif signature_name == _TT_SUMMARY_MAX:
                    signature_result_tensor = _show_max(tensor, cast_to_f32=False)
                elif signature_name == _TT_SUMMARY_MAX_ABS:
                    signature_result_tensor = _show_max_abs(tensor, cast_to_f32=False)
                elif signature_name == _TT_SUMMARY_MIN:
                    signature_result_tensor = _show_min(tensor, cast_to_f32=False)
                elif signature_name == _TT_SUMMARY_SPARSITY:
                    signature_result_tensor = _show_sparsity(tensor)
                elif signature_name == _TT_SUMMARY_SIZE:
                    signature_result_tensor = _show_size(tensor)
                elif signature_name == _TT_SUMMARY_MEAN:
                    signature_result_tensor = mean
                elif signature_name == _TT_SUMMARY_VAR:
                    signature_result_tensor = variance
                else:
                    raise ValueError('Unknown signature type :%s.' % signature_name)
                result_dict[signature_name] = signature_result_tensor
            return result_dict
        raise RuntimeError('Unsupported signature for trace mode %s.' % self._parameters.trace_mode)

    def _make_tensor_trace_fun(self, tensor_name, tensor_trace_order):
        """Makes the tensor tracing function called by outside compilation.

    Args:
      tensor_name: name of the tensor being traced.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
    Returns:
      A function to be passed as the first argument to outside compilation.

    Raises:
      RuntimeError: If the trace mode is invalid.
    """

        def _print_tensor(tensor_name, num_elements, tensor, output_tensor):
            """Prints a tensor value to a file.

      Args:
        tensor_name: name of the tensor being traced.
        num_elements: number of elements to print (-1 means print all).
        tensor: the tensor needs to be returned.
        output_tensor: the tensor needs to be printed.

      Returns:
        The same tensor passed via the "tensor" argument.

      Raises:
        ValueError: If tensor_name is not already in
                    tensor_trace_order.tensorname_to_cache_idx.
      """
            if self._parameters.is_brief_mode():
                if tensor_name not in tensor_trace_order.tensorname_to_cache_idx:
                    raise ValueError('Tensor %s with name %s is not in the tensorname_to_cache_idx' % (tensor, tensor_name))
                msg = '%d' % tensor_trace_order.tensorname_to_cache_idx[tensor_name]
            else:
                msg = '"%s"' % tensor_name
            if self._parameters.trace_dir:
                output_path = os.path.join(self._parameters.trace_dir, _TRACE_FILE_NAME + self._get_outfile_suffix())
                output_stream = _OUTPUT_STREAM_ESCAPE + output_path
            else:
                output_stream = sys.stderr
            return logging_ops.print_v2(msg, array_ops.shape(output_tensor), '@', self._replica_id, '\n', output_tensor, '\n', summarize=num_elements, output_stream=output_stream)

        def _show_part_tensor(tensor):
            """Trace function for printing part of the tensor."""
            return _print_tensor(tensor_name, _TRACE_MODE_PART_TENSOR_SIZE, tensor, tensor)

        def _show_full_tensor(tensor):
            """Trace function for printing the entire tensor."""
            return _print_tensor(tensor_name, -1, tensor, tensor)
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_PART_TENSOR:
            return _show_part_tensor
        if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_NAN_INF, tensor_tracer_flags.TRACE_MODE_NORM, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR, tensor_tracer_flags.TRACE_MODE_MAX_ABS, tensor_tracer_flags.TRACE_MODE_SUMMARY, tensor_tracer_flags.TRACE_MODE_HISTORY):
            return _show_full_tensor
        raise RuntimeError('Full tensor support is not available with trace mode %s' % self._parameters.trace_mode)

    def _is_in_control_flow(self, op):
        """Returns true if the given op is inside a tf.cond or in tf.while_loop.

    Args:
      op: A tensorflow op that should be checked whether in control flow or not.
    Returns:
      A boolean value whether the op is in control flow or not.
    """
        return control_flow_util.IsInCond(op)

    def _is_in_outmost_while_loop(self, op):
        """Returns true if the op is at the same level with the training loop.

    Returns false if the op is in an inner while loop or if it is outside of the
    training loop.
    Args:
      op: tf.Operation

    Returns:
      A boolean.
    """
        ctxt = self._get_op_control_flow_context(op)
        outer_while_context = control_flow_util.GetContainingWhileContext(ctxt)
        return outer_while_context == control_flow_util.GetContainingWhileContext(self._outmost_context)

    def _should_trace_in_control_flow(self):
        """Returns false incase it is not safe to trace ops in tf.cond or tf.while_loop."""
        if self._use_temp_cache():
            return False
        elif self._tt_config.device_type == _DEVICE_TYPE_TPU:
            return self._use_tensor_values_cache() or self._use_tensor_buffer()
        return True

    def _skip_op(self, op_id, op, ops_in_exec_path, report_handler):
        """Returns True if we should not trace Op.

    Args:
      op_id: Topological index of the op.
      op: tf.Operation
      ops_in_exec_path: Set of operations that are in the execution path.
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      True if the op should not be traced, false otherwise.
    """
        if TensorTracer.while_loop_op(op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_WHILELOOP_OP))
            return True
        if TensorTracer.control_flow_op(op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_CONTROLFLOW_OP))
            return True
        if TensorTracer.unsafe_op(op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_UNSAFE_OP))
            return True
        if TensorTracer.device_mismatch(self._tt_config.device_type, op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_DEVICE_MISMATCH))
            return True
        if op not in ops_in_exec_path:
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_NOT_EXECUTED))
            return True
        if self._is_in_control_flow(op) or not self._is_in_outmost_while_loop(op):
            if not self._should_trace_in_control_flow():
                report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_IN_CONTROL_FLOW))
                return True
        if self._is_user_included_op(op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_USER_INCLUDED))
            if tensor_tracer_flags.TT_CHECK_FILTER.value:
                logging.info('USER_INCLUDED op %s', op.name)
            return False
        if not self._inside_op_range(op_id):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_OUTSIDE_OP_RANGE))
            return True
        if not self._is_interesting_op(op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_LESS_INTERESTING_OP))
            return True
        if self._is_user_excluded_op(op):
            report_handler.instrument_op(op, TensorTracer.reason(op_id, _REASON_USER_EXCLUDED))
            if tensor_tracer_flags.TT_CHECK_FILTER.value:
                logging.info('USER_EXCLUDED op %s', op.name)
            return True
        return False

    def _skip_tensor(self, op_id, out_tensor, report_handler):
        """Returns True if we should not trace out_tensor.

    Args:
      op_id: Topological index of the op producing tensor.
      out_tensor: tf.Tensor
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      True if the tensor should not be traced, false otherwise.
    """
        non_numeric_tensor_types = set([dtypes.variant, dtypes.resource, dtypes.string])
        if out_tensor.dtype in non_numeric_tensor_types:
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_NON_NUMERIC_TENSOR))
            return True
        if [consumer for consumer in out_tensor.consumers() if TensorTracer.while_loop_op(consumer)]:
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_FEEDS_WHILELOOP_OP))
            return True
        if self._is_user_included_op(out_tensor.op):
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_USER_INCLUDED))
            if tensor_tracer_flags.TT_CHECK_FILTER.value:
                logging.info('USER_INCLUDED tensor %s', out_tensor.name)
            return False
        if self._is_user_excluded_op(out_tensor.op):
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_USER_EXCLUDED))
            if tensor_tracer_flags.TT_CHECK_FILTER.value:
                logging.info('USER_EXCLUDED tensor %s', out_tensor.name)
            return True
        if not out_tensor.get_shape().is_fully_defined():
            if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_NAN_INF, tensor_tracer_flags.TRACE_MODE_NORM, tensor_tracer_flags.TRACE_MODE_HISTORY, tensor_tracer_flags.TRACE_MODE_MAX_ABS, tensor_tracer_flags.TRACE_MODE_SUMMARY):
                report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_TENSOR_GET_TRACED))
                return False
            else:
                report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_DYNAMIC_SHAPE))
                return True
        rank = len(out_tensor.shape)
        if rank < 1:
            if self._parameters.trace_scalar_ops:
                if TensorTracer.unsafe_scalar_trace(out_tensor.op):
                    report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_UNSAFE_SCALAR))
                    return True
                else:
                    report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_SCALAR_GET_TRACED))
                    return False
            else:
                report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_SKIP_SCALAR))
                return True
        else:
            report_handler.instrument_tensor(out_tensor, TensorTracer.reason(op_id, _REASON_TENSOR_GET_TRACED))
            return False

    def _filter_execution_path_operations(self, operations, fetches):
        """Returns the set of ops in the execution path to compute given fetches."""
        if fetches is None:
            return set(operations)
        if not isinstance(fetches, (list, tuple)):
            fetches = [fetches]
        op_fetches = []
        for fetch in fetches:
            if isinstance(fetch, ops.Operation):
                op_fetches.append(fetch)
            elif isinstance(fetch, tensor_lib.Tensor):
                op_fetches.append(fetch.op)
            else:
                raise RuntimeError('Given fetch:%s is neither a tensor nor an op.' % fetch)
        execution_path_operations = set(op_fetches)
        traverse_stack = list(op_fetches)
        while True:
            if not traverse_stack:
                break
            head_op = traverse_stack.pop()
            input_ops = [tensor_input.op for tensor_input in head_op.inputs]
            input_ops.extend(head_op.control_inputs)
            for input_op in input_ops:
                if input_op not in execution_path_operations:
                    if TensorTracer.loop_cond_op(input_op):
                        continue
                    execution_path_operations.add(input_op)
                    traverse_stack.append(input_op)
        return execution_path_operations

    def _determine_and_instrument_traced_tensors(self, graph_order, ops_in_exec_path, tensor_trace_points, report_handler):
        """Determines the tensors to trace and instruments the trace details.

    Args:
      graph_order: graph_order tuple containing graph (tf.graph), operations
        (list of operations), op_to_idx (op id mapping), (tensors) list of
        tensors, tensor_to_idx (tensor id mapping), contains_cycle (whether
        there is a cycle in the graph), topological_order_or_cycle (list of ops
        in topological order or list of ops creating a cycle).
      ops_in_exec_path: Set of ops in the execution path.
      tensor_trace_points: Collection of programatic tensor trace points.
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      List of tensors to be traced.
    """
        traced_tensors = []
        checkpoint_operations = set([tensor.op for tensor, _ in tensor_trace_points])
        for op_id, op in enumerate(graph_order.operations):
            if checkpoint_operations and op not in checkpoint_operations:
                continue
            if self._skip_op(op_id, op, ops_in_exec_path, report_handler):
                continue
            for i in range(len(op.outputs)):
                out_tensor = op.outputs[i]
                if not self._skip_tensor(op_id, out_tensor, report_handler):
                    traced_tensors.append(out_tensor)
        return traced_tensors

    def _check_trace_files(self):
        """Checks if any requirements for trace files are satisfied."""
        if not self._parameters.trace_dir:
            return
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
            return
        if not gfile.Exists(self._parameters.trace_dir):
            file_io.recursive_create_dir(self._parameters.trace_dir)
            if not gfile.Exists(self._parameters.trace_dir):
                raise RuntimeError('Failed to create trace directory at %s' % self._parameters.trace_dir)

    def _create_temp_cache(self, num_traced_tensors, num_signatures, graph):
        """Creates a temporary cache with the given dimensions.

    Fills the self._temp_cache_var with num_traced_tensors tf.constant() ops
    that have shape of [num_signatures].
    Args:
      num_traced_tensors: Int, denoting total number of traced tensors.
      num_signatures: Int, denoting the number of statistics collected per
        tensors.
      graph: TensorFlow graph.
    """
        init_value = constant_op.constant(_COMPACT_TRACE_ENTRY_INIT_VALUE, dtype=dtypes.float32, shape=[num_signatures])
        self._temp_cache_var[graph] = [init_value for _ in range(num_traced_tensors)]

    def _determine_trace_and_create_report(self, graph, ops_in_exec_path, graph_summary_tag):
        """Work needs to be done prior to TPU or CPU tracing.

    Args:
      graph: tf.graph
      ops_in_exec_path: Set of operations in the execution path.
      graph_summary_tag: the summary tag name for the given graph.
    Returns:
      An instance of tensor_tracer_report.TensorTraceOrder, containing list of
      tensors to be traced with their topological order information.
    Raises:
      RuntimeError: If opname filtering is incorrectly set.
    """
        self._check_trace_files()
        graph_order = tensor_tracer_report.sort_tensors_and_ops(graph)
        tensor_trace_points = graph.get_collection(_TENSOR_TRACER_COLLECTION)
        report_handler = tensor_tracer_report.TTReportHandle()
        traced_tensors = self._determine_and_instrument_traced_tensors(graph_order, ops_in_exec_path, tensor_trace_points, report_handler)
        logging.info('TensorTracer is tracing %d tensors.', len(traced_tensors))
        if traced_tensors and tensor_tracer_flags.TT_CHECK_FILTER.value:
            raise RuntimeError('Verify ops being traced by tensor tracer.')
        tensor_trace_order = tensor_tracer_report.TensorTraceOrder(graph_order, traced_tensors)
        num_signatures = self._num_signature_dimensions()
        if num_signatures and self._use_tensor_values_cache():
            if self._use_temp_cache():
                self._create_temp_cache(len(traced_tensors), num_signatures, graph)
            else:
                self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG, graph, [len(traced_tensors), num_signatures])
                if self._parameters.trace_mode in tensor_tracer_flags.TRACE_MODE_HISTORY:
                    self._create_or_get_tensor_history_values_cache(_TT_SUMMARY_TAG, graph, [len(traced_tensors), num_signatures])
        if self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_SUMMARY, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY):
            self._report_proto = report_handler.create_report_proto(self._tt_config, self._parameters, tensor_trace_order, tensor_trace_points, self._signature_types())
            if self._parameters.use_fingerprint_subdir:
                self._parameters.trace_dir = os.path.join(self._parameters.trace_dir, self._report_proto.fingerprint)
                logging.info('TensorTracer updating trace_dir to %s', self._parameters.trace_dir)
            self._report_proto_path = report_handler.report_proto_path(self._parameters.trace_dir, graph_summary_tag)
            if self._parameters.report_file_path != _SKIP_REPORT_FILE:
                report_handler.write_report_proto(self._report_proto_path, self._report_proto, self._parameters)
        elif self._parameters.trace_mode not in tensor_tracer_flags.TRACE_MODE_HISTORY:
            report_handler.create_report(self._tt_config, self._parameters, tensor_trace_order, tensor_trace_points)
        return tensor_trace_order

    def _create_host_call(self):
        return self._parameters.trace_mode in (tensor_tracer_flags.TRACE_MODE_SUMMARY, tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY)

    def _inspect_summary_cache(self, cache, replica_id, step_num, output_stream, tensor_trace_order):
        """Generates a print operation to print trace inspection.

    Args:
      cache: Tensor storing the trace results for the step.
      replica_id: Tensor storing the replica id of the running core.
      step_num: Step number.
      output_stream: Where to print the outputs, e.g., file path, or sys.stderr.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.

    Returns:
      The Op to flush the cache to file.
    """

        def _inspect_tensor(tensor):
            """Returns the text to be printed for inspection output."""
            if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
                return cond.cond(math_ops.greater(tensor, 0.0), lambda: 'has NaNs/Infs!', lambda: 'has no NaNs or Infs.')
            else:
                return tensor
        if not tensor_trace_order.traced_tensors:
            logging.warn('Inspect mode has no tensors in the cache to check.')
            return control_flow_ops.no_op
        if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
            step_has_nan_or_inf = math_ops.greater(math_ops.reduce_sum(cache), 0.0)
        else:
            step_has_nan_or_inf = math_ops.reduce_any(gen_math_ops.logical_or(gen_math_ops.is_nan(cache), gen_math_ops.is_inf(cache)))
        step_error_message = cond.cond(step_has_nan_or_inf, lambda: 'NaNs or Infs in the step!', lambda: 'No numerical issues have been found for the step.')
        if self._parameters.collect_summary_per_core:
            stats = ['\n\n', 'core:', replica_id, ',', 'step:', step_num, '-->', step_error_message, 'Printing tensors for mode:%s...' % self._parameters.trace_mode]
        else:
            stats = ['\n\n', 'step:', step_num, '-->', step_error_message, 'Printing tensors for mode:%s...' % self._parameters.trace_mode]
        for tensor_name, cache_idx in sorted(tensor_trace_order.tensorname_to_cache_idx.items(), key=lambda item: item[1]):
            if self._parameters.collect_summary_per_core:
                stats.extend(['\n', 'core:', replica_id, ',', 'step:', step_num, ',', tensor_name, '-->', _inspect_tensor(cache[cache_idx, 0])])
            else:
                stats.extend(['\n', 'step:', step_num, ',', tensor_name, '-->', _inspect_tensor(cache[cache_idx, 0])])
        return logging_ops.print_v2(*stats, summarize=-1, output_stream=output_stream)

    def _inspect_history_cache(self, cache, replica_id, step_num, tensor_trace_order):
        """Generates a conditional print operation to log differences in tensor values.

    Args:
      cache: Tensor storing the trace results for the step.
      replica_id: Tensor storing the replica id of the running core.
      step_num: Step number.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.

    Returns:
      The Op to flush the cache to file.
    """
        if not tensor_trace_order.traced_tensors:
            logging.warn('TT history mode has no tensors in the cache to check.')
            return control_flow_ops.no_op
        stats = ['\n\n', 'core:', replica_id, ',', 'step:', step_num]
        diffs = []
        for tensor_name, cache_idx in sorted(tensor_trace_order.tensorname_to_cache_idx.items(), key=lambda item: item[1]):
            tensor_to_write = cache[cache_idx, 0]
            snapshot_variable = self._create_or_get_tensor_history_values_cache(tensor_to_write.name, tensor_to_write.op.graph, tensor_to_write.shape.as_list(), tensor_to_write.dtype)
            with ops.control_dependencies([snapshot_variable]):
                old_value = state_ops.assign_add(snapshot_variable, 0.0)
            with ops.control_dependencies([old_value]):
                new_value = math_ops.cast(tensor_to_write, dtypes.float32)
                delta = math_ops.abs(math_ops.subtract(old_value, new_value))
                updated = state_ops.assign(snapshot_variable, new_value)
                diffs.append(delta)
            with ops.control_dependencies([updated]):
                new_value_from_var = state_ops.assign_add(snapshot_variable, 0.0)
            stats.extend(['\n', 'core:', replica_id, ',', 'step:', step_num, ',', tensor_name, '-->', old_value, new_value_from_var, delta])
        diff_stack = array_ops_stack.stack(diffs)
        step_max = math_ops.reduce_max(diff_stack)
        return cond.cond(math_ops.greater(step_max, tensor_tracer_flags.DELTA_THRESHOLD.value), lambda: logging_ops.print_v2(*stats, summarize=-1), lambda: control_flow_ops.no_op())

    def _get_outfile_suffix(self):
        if remote_utils.is_remote_path(self._parameters.trace_dir):
            return remote_utils.get_appendable_file_encoding()
        else:
            return ''

    def _generate_flush_cache_op(self, num_replicas, on_tpu, tensor_trace_order, graph):
        """Generates an Op that will flush the cache to file.

    Args:
      num_replicas: total number of replicas.
      on_tpu: if the graph is executed on TPU.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
      graph: TensorFlow graph.

    Returns:
      The Op to flush the cache to file.
    """

        def _flush_fun(cache, replica_id, step_num):
            """Flushes the cache to a file corresponding to replica_id."""

            def _f(file_index):
                """Generates a func that flushes the cache to a file."""

                def _print_cache():
                    """Flushes the cache to a file."""
                    replica_str = '%d' % file_index
                    if self._parameters.trace_dir:
                        output_path = os.path.join(self._parameters.trace_dir, _COMPACT_TRACE_FILE_PREFIX) + replica_str + self._get_outfile_suffix()
                        output_stream = _OUTPUT_STREAM_ESCAPE + output_path
                    else:
                        output_stream = sys.stderr
                    new_step_line = _REPLICA_ID_TAG + replica_str
                    print_ops = []
                    if self._parameters.inspect_trace:
                        if self._num_signature_dimensions() > 1:
                            raise ValueError('Inspecting multi signatures are not supported.')
                        if self._parameters.trace_mode in tensor_tracer_flags.TRACE_MODE_HISTORY:
                            print_ops.append(self._inspect_history_cache(cache=cache, replica_id=replica_id, step_num=step_num, tensor_trace_order=tensor_trace_order))
                        else:
                            print_ops.append(self._inspect_summary_cache(cache=cache, replica_id=replica_id, step_num=step_num, output_stream=output_stream, tensor_trace_order=tensor_trace_order))
                    else:
                        for i in range(self._num_signature_dimensions()):
                            print_ops.append(logging_ops.print_v2(new_step_line, '\n', cache[:, i], '\n', summarize=-1, output_stream=output_stream))
                    with ops.control_dependencies(print_ops):
                        return constant_op.constant(0).op
                return _print_cache

            def _eq(file_index):
                return math_ops.equal(replica_id, file_index)
            flush_op_cases = {}
            flush_op_cases[_eq(0)] = _f(0)
            for i in range(1, num_replicas):
                if on_tpu and (not self._parameters.collect_summary_per_core):
                    flush_op_cases[_eq(i)] = control_flow_ops.no_op
                else:
                    flush_op_cases[_eq(i)] = _f(i)
            return control_flow_case.case(flush_op_cases, exclusive=True)
        cache = self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG, graph)
        if self._use_temp_cache():
            cache_val = cache
        else:
            cache_val = cache.value()
        if on_tpu:
            if not self._parameters.collect_summary_per_core:
                cache_val = self.merge_caches_on_tpu(cache_val)
                cache_val = self.aggregate_global_cache(cache_val)[0]
            flush_op = tpu_replication.outside_compilation(_flush_fun, cache_val, self._replica_id, array_ops.identity(training_util.get_or_create_global_step()))
        else:
            global_step = training_util.get_or_create_global_step()
            flush_op = _flush_fun(cache_val, self._replica_id, global_step)
        if self._use_temp_cache():
            with ops.control_dependencies([flush_op]):
                return constant_op.constant(0).op
        else:
            with ops.control_dependencies([flush_op]):
                reset_value = constant_op.constant(_COMPACT_TRACE_ENTRY_INIT_VALUE, dtype=cache.dtype, shape=cache.shape)
                assign_op = state_ops.assign(cache, reset_value).op
                with ops.control_dependencies([assign_op]):
                    return constant_op.constant(0).op

    def _flush_tensor_values_cache(self, tensor_fetches, op_fetches, on_tpu, tensor_trace_order, graph):
        """Flushes the intermediate tensor values in the graph to the cache.

    Args:
      tensor_fetches: list of tensor results returned by the model_fn.
      op_fetches: list of ops that are returned by the model_fn, e.g., train_op.
      on_tpu: if the graph is executed on TPU.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
      graph: TensorFlow graph.

    Returns:
      An identical copy of tensor_fetches.
    """
        if not tensor_trace_order.traced_tensors:
            logging.warn('No tensor values being traced. No flush cache op added.')
            return tensor_fetches
        with ops.control_dependencies(op_fetches + [tensor.op for tensor in tensor_fetches]):
            flush_cache_op = self._generate_flush_cache_op(self._tt_config.num_replicas, on_tpu, tensor_trace_order, graph)
            return control_flow_ops.tuple(tensor_fetches, control_inputs=[flush_cache_op])

    def _process_tensor_fetches(self, tensor_fetches):
        """Check that tensor_fetches is not empty and have valid tensors."""
        if tensor_fetches is None:
            raise RuntimeError('tensor_fetches provided to tensor_tracer cannot be None.')
        if not isinstance(tensor_fetches, (list, tuple)):
            tensor_fetches = [tensor_fetches]
        elif not tensor_fetches:
            raise RuntimeError('tensor_fetches provided to tensor_tracer cannot be empty list.')
        fetches = []
        for fetch in tensor_fetches:
            if isinstance(fetch, tensor_lib.Tensor):
                fetches.append(fetch)
            else:
                raise RuntimeError('Given tensor_fetch:%s is not a tensor.' % fetch)
        return fetches

    def _process_op_fetches(self, op_fetches):
        """Check that op_fetches have valid ops."""
        if op_fetches is None:
            return []
        if not isinstance(op_fetches, (list, tuple)):
            op_fetches = [op_fetches]
        fetches = []
        for fetch in op_fetches:
            if isinstance(fetch, ops.Operation):
                fetches.append(fetch)
            elif isinstance(fetch, tensor_lib.Tensor):
                fetches.append(fetch.op)
            else:
                logging.warning('Ignoring the given op_fetch:%s, which is not an op.' % fetch)
        return fetches

    def _convert_fetches_to_input_format(self, input_fetches, current_fetches):
        """Changes current_fetches' format, so that it matches input_fetches."""
        if isinstance(input_fetches, tensor_lib.Tensor):
            if len(current_fetches) != 1:
                raise RuntimeError('Tensor tracer input/output fetches do not match.')
            return current_fetches[0]
        elif len(current_fetches) != len(current_fetches):
            raise RuntimeError('Tensor tracer input/output fetches do not match.')
        elif isinstance(input_fetches, tuple):
            return tuple(current_fetches)
        else:
            return current_fetches

    def _get_op_control_flow_context(self, op):
        """Returns the control flow of the given op.

    Args:
      op: tf.Operation for which the control flow context is requested.
    Returns:
      op_control_flow_context: which the is control flow context of the given
      op. If the operation type is LoopExit, returns the outer control flow
      context.
    """
        op_control_flow_context = op._control_flow_context
        if control_flow_util.IsLoopExit(op):
            op_control_flow_context = op_control_flow_context.outer_context
        return op_control_flow_context

    def merge_caches_on_tpu(self, local_tpu_cache_tensor):
        """Merges the given caches on tpu.

    Args:
      local_tpu_cache_tensor: A local tensor that needs to be merged
        by concanting data from other tpu cores.
    Returns:
      A merged tf.Tensor.
    """
        x = array_ops.broadcast_to(local_tpu_cache_tensor, shape=[self._tt_config.num_replicas] + local_tpu_cache_tensor.shape.as_list())
        if tensor_tracer_flags.TT_SINGLE_CORE_SUMMARIES.value:
            return x
        return tpu_ops.all_to_all(x, concat_dimension=0, split_dimension=0, split_count=self._tt_config.num_replicas, group_assignment=[list(range(self._tt_config.num_replicas))])

    def aggregate_global_cache(self, global_tt_summary_cache):
        """Merges the given caches on tpu.

    Args:
      global_tt_summary_cache: The global tensor tracer summary cache tensor
        with shape (num_cores, num_traced_tensors, num_traced_signatures). First
        dimension corresponds to core_id, where global_tpu_cache_tensor[i]
        correspond to the local cache from core-i.
    Returns:
      An aggregated tf.Tensor.
    Raises:
      RuntimeError: if there is no aggregate function defined for a signature.
    """
        agg_fn_map = self._parameters.get_signature_to_agg_fn_map()
        signature_idx_map = self._signature_types()
        aggregation_result = []
        for signature, idx in sorted(signature_idx_map.items(), key=operator.itemgetter(1)):
            if signature not in agg_fn_map:
                raise RuntimeError('No aggregation function is defined for signature %s.' % signature)
            signature_tensor = global_tt_summary_cache[:, :, idx]
            agg_fn = agg_fn_map[signature]
            agg_tensor = agg_fn(signature_tensor, axis=0)
            aggregation_result.append(agg_tensor)
        merged_signatures = array_ops_stack.stack(aggregation_result)
        transposed_signatures = array_ops.transpose(merged_signatures)
        return array_ops.expand_dims(transposed_signatures, axis=0)

    def _prepare_host_call_fn(self, processed_t_fetches, op_fetches, graph, graph_summary_tag):
        """Creates a host call function that will write the cache as tb summary.

    Args:
      processed_t_fetches: List of tensor provided to session.run.
      op_fetches: List of operations provided to session.run.
      graph: TensorFlow graph.
      graph_summary_tag: the summary_tag name for the given graph.
    Raises:
      ValueError if trace_dir is not set.
    """
        if self._parameters.trace_dir is None:
            raise ValueError('Provide a trace_dir for tensor tracer in summary mode. --trace_dir=/model/dir')

        def _write_cache(step, event_file_suffix=None, **kwargs):
            """Writes the given caches as tensor summary.

      Args:
        step: Step tensor with dimension [num_cores].
        event_file_suffix: Event filename suffix tensor.
        **kwargs: The dictionary of tensors that needs to be written as
          summaries. Key and value pairs within kwargs correspond to the tag
          name, and tensor content that will be written using summary.write.
          The trace_modes that use this function are:
            - summary: In summary mode, kwargs includes a single (tag, content)
            pair which are, _TT_SUMMARY_TAG and a tf.float32 signature_cache
            variable. The dimension of the signature_cache is:
              num_cores x num_traced_tensors x num_signatures.
            - full_tensor_summary: kwargs will include all traced tensors. Tag
            and content correspond to the name of the tensor, and its actual
            content.
      Returns:
        A tf.Operation that needs to be executed for the host call dependencies.
      """
            file_suffix = _TT_EVENT_FILE_SUFFIX
            if event_file_suffix is not None:
                file_suffix = string_ops.string_join([file_suffix, event_file_suffix], separator='.')
            summary_write_ops = []
            summary_writer = summary.create_file_writer_v2(self._parameters.trace_dir, filename_suffix=file_suffix, max_queue=_TT_SUMMARY_MAX_QUEUE)
            graph.add_to_collection(TENSOR_TRACER_SUMMARY_COLLECTION, summary_writer)
            step_value = step[0]
            dt = step_value.dtype
            if dt.__ne__(dtypes.int64) and dt.__ne__(dtypes.uint64) and dt.__ne__(dtypes.float64):
                step_value = math_ops.cast(step_value, dtypes.int64)
            with summary_writer.as_default():
                summary_metadata = summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name=_TT_TENSORBOARD_PLUGIN_NAME))
                for key, value in kwargs.items():
                    if not self._parameters.collect_summary_per_core:
                        if key == _TT_SUMMARY_TAG and value.shape.as_list()[0] != 1:
                            value = self.aggregate_global_cache(value)
                    with ops.control_dependencies([summary_writer.init()]):
                        summary_write_ops.append(summary.write(_TT_SUMMARY_TAG + '/' + key + '.' + graph_summary_tag, value, metadata=summary_metadata, step=step_value))
            return control_flow_ops.group(summary_write_ops)
        global_step = training_util.get_or_create_global_step()
        step = array_ops.reshape(global_step, [1])
        self._host_call_fn = {}
        host_call_deps = op_fetches + [tensor.op for tensor in processed_t_fetches]
        caches_to_write = {}
        with ops.control_dependencies(host_call_deps):
            all_caches = self._cache_variable_for_graph(graph)
            for cache_name, cache_variable in all_caches.items():
                new_cache_shape = [1]
                new_cache_shape.extend(cache_variable.shape.as_list())
                cache = array_ops.reshape(cache_variable, new_cache_shape)
                caches_to_write[cache_name] = cache
        caches_to_write['step'] = step
        self._host_call_fn[_TT_HOSTCALL_KEY] = (_write_cache, caches_to_write)

    def host_call_deps_and_fn(self):
        return self._host_call_fn

    def get_traced_op_names(self):
        """Returns the set of traced op names."""
        return self._traced_op_names

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

    def trace_tpu(self, graph, tensor_fetches, op_fetches=None, num_replicas=None, num_replicas_per_host=None, num_hosts=None):
        """Traces the tensors generated by TPU Ops in a TF graph.

    Args:
      graph: the graph of Ops executed on the TPU.
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.
      num_replicas: number of replicas used on the TPU.
      num_replicas_per_host: number of replicas per TPU host.
      num_hosts: total number of TPU hosts.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    """
        if isinstance(graph, func_graph.FuncGraph) or isinstance(graph, function._FuncGraph):
            logging.warning('Tensor Tracer is not supported for tracing FuncGraphs. Ignoring tracing.')
            return tensor_fetches
        if graph in TensorTracer._traced_graphs:
            logging.warning('Graph is already rewritten with tensor tracer, ignoring multiple calls.')
            return tensor_fetches
        else:
            TensorTracer._traced_graphs.add(graph)
        self._parameters = tensor_tracer_flags.TTParameters()
        self._tt_config.device_type = _DEVICE_TYPE_TPU
        self._tt_config.num_replicas = num_replicas
        self._tt_config.num_replicas_per_host = num_replicas_per_host
        self._tt_config.num_hosts = num_hosts
        if self._tt_config.num_replicas is not None:
            if self._tt_config.num_replicas_per_host is None:
                self._tt_config.num_replicas_per_host = 8
            if self._tt_config.num_hosts is None:
                self._tt_config.num_hosts = num_replicas // self._tt_config.num_replicas_per_host + (num_replicas % self._tt_config.num_replicas_per_host > 0)
        if self._parameters.graph_dump_path:
            graph_io.write_graph(graph, self._parameters.graph_dump_path, 'graph_before_tt.pbtxt')
        with graph.as_default():
            self._add_replica_id_to_graph()
            tensor_fetches = self._trace_execution(graph, tensor_fetches, op_fetches, on_tpu=True)
        if self._parameters.graph_dump_path:
            graph_io.write_graph(graph, self._parameters.graph_dump_path, 'graph_after_tt.pbtxt')
        return tensor_fetches

    def trace_cpu(self, graph, tensor_fetches, op_fetches=None):
        """Traces the tensors generated by CPU Ops in a TF graph.

    Args:
      graph: the graph of Ops executed on the CPU.
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    """
        if isinstance(graph, func_graph.FuncGraph) or isinstance(graph, function._FuncGraph):
            logging.warning('Tensor Tracer is not supported for tracing FuncGraphs. Ignoring tracing.')
            return tensor_fetches
        if graph in TensorTracer._traced_graphs:
            logging.warning('Graph is already rewritten with tensor tracer, ignoring multiple calls.')
            return tensor_fetches
        else:
            TensorTracer._traced_graphs.add(graph)
        self._parameters = tensor_tracer_flags.TTParameters()
        self._tt_config.device_type = _DEVICE_TYPE_CPU
        self._tt_config.num_replicas = 1
        self._tt_config.num_replicas_per_host = 1
        self._tt_config.num_hosts = 1
        self._replica_id = 0
        if self._parameters.graph_dump_path:
            graph_io.write_graph(graph, self._parameters.graph_dump_path, 'graph_before_tt.pbtxt')
        with graph.as_default():
            tensor_fetches = self._trace_execution(graph, tensor_fetches, op_fetches, on_tpu=False)
        if self._parameters.graph_dump_path:
            graph_io.write_graph(graph, self._parameters.graph_dump_path, 'graph_after_tt.pbtxt')
        return tensor_fetches