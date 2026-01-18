from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
def _invoke_input_fn_and_record_structure(self):
    """Deploys the input pipeline and record input structure."""
    enqueue_ops = []
    infeed_queues = []
    all_dataset_initializers = []
    num_hosts = self._ctx.num_hosts
    tpu_host_placement_fn = self._ctx.tpu_host_placement_function
    run_infeed_loop_on_coordinator = True
    if self._sharded_per_core:
        for host_id in range(num_hosts):
            host_device = tpu_host_placement_fn(host_id=host_id)
            with tf.compat.v1.device(host_device):
                with ops.name_scope('input_pipeline_task%d' % host_id):
                    enqueue_ops_fn, captured_infeed_queue = generate_per_core_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, host_device, host_id)
                    if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
                        run_infeed_loop_on_coordinator = False
                        enqueue_ops.append(_wrap_computation_in_while_loop(device=host_device, op_fn=enqueue_ops_fn))
                    else:
                        enqueue_ops.append(enqueue_ops_fn())
                    infeed_queues.append(captured_infeed_queue.get())
    elif self._ctx.is_input_broadcast_with_iterators():
        host_device = tpu_host_placement_fn(host_id=0)
        enqueue_ops_fn, captured_infeed_queue, dataset_initializer = generate_broadcast_enqueue_ops_fn(self._ctx, self._input_fn, self._inputs_structure_recorder, num_hosts)
        if dataset_initializer:
            all_dataset_initializers.append(dataset_initializer)
            run_infeed_loop_on_coordinator = False
            wrap_fn = _wrap_computation_in_while_loop if self._ctx.mode != model_fn_lib.ModeKeys.PREDICT else _wrap_computation_in_while_loop_with_stopping_signals
            enqueue_ops.append(wrap_fn(device=host_device, op_fn=enqueue_ops_fn))
        else:
            enqueue_ops.append(enqueue_ops_fn())
        infeed_queues.append(captured_infeed_queue.get())
    else:
        host_id_with_invocation_id_pair = []
        if not self._ctx.is_replica_across_hosts():
            for host_id in range(num_hosts):
                invocation_index = host_id
                host_id_with_invocation_id_pair.append((host_id, invocation_index))
        else:
            for replica_id in xrange(self._ctx.num_replicas):
                invocation_index = replica_id
                host_device, _ = self._ctx.device_for_replica(replica_id)
                host_id = int(host_device.split('/task:')[1].split('/device:')[0])
                host_id_with_invocation_id_pair.append((host_id, invocation_index))
        for host_id, invocation_index in host_id_with_invocation_id_pair:
            host_device = tpu_host_placement_fn(host_id=host_id)
            with tf.compat.v1.device(host_device):
                with ops.name_scope('input_pipeline_task%d' % host_id):
                    if self._ctx.is_input_per_host_with_iterators():
                        enqueue_ops_fn, captured_infeed_queue, dataset_initializer = generate_per_host_v2_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, host_device, host_id, invocation_index)
                    else:
                        enqueue_ops_fn, captured_infeed_queue, dataset_initializer = generate_per_host_enqueue_ops_fn_for_host(self._ctx, self._input_fn, self._inputs_structure_recorder, self._batch_axis, host_device, host_id)
                    if dataset_initializer:
                        all_dataset_initializers.append(dataset_initializer)
                        run_infeed_loop_on_coordinator = False
                        wrap_fn = _wrap_computation_in_while_loop if self._ctx.mode != model_fn_lib.ModeKeys.PREDICT else _wrap_computation_in_while_loop_with_stopping_signals
                        enqueue_ops.append(wrap_fn(device=host_device, op_fn=enqueue_ops_fn))
                    else:
                        enqueue_ops.append(enqueue_ops_fn())
                    infeed_queues.append(captured_infeed_queue.get())
    self._infeed_queue = infeed_queues[0]
    return (enqueue_ops, [util_lib.MultiHostDatasetInitializerHook(all_dataset_initializers)], run_infeed_loop_on_coordinator)