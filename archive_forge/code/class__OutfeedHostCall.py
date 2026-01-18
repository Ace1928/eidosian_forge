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
class _OutfeedHostCall(object):
    """Support for `eval_metrics` and `host_call` in TPUEstimatorSpec."""

    def __init__(self, ctx, outfeed_every_n_steps=1):
        self._ctx = ctx
        self._names = []
        self._host_fns = {}
        self._tensor_keys = collections.defaultdict(list)
        self._tensors = collections.defaultdict(list)
        self._tensor_dtypes = collections.defaultdict(list)
        self._tensor_shapes = collections.defaultdict(list)
        self._outfeed_every_n_steps = outfeed_every_n_steps

    @staticmethod
    def validate(host_calls):
        """Validates the `eval_metrics` and `host_call` in `TPUEstimatorSpec`."""
        for name, host_call in host_calls.items():
            if not isinstance(host_call, (tuple, list)):
                raise ValueError('{} should be tuple or list'.format(name))
            if len(host_call) != 2:
                raise ValueError('{} should have two elements.'.format(name))
            if not callable(host_call[0]):
                raise TypeError('{}[0] should be callable.'.format(name))
            if not isinstance(host_call[1], (tuple, list, dict)):
                raise ValueError('{}[1] should be tuple or list, or dict.'.format(name))
            if isinstance(host_call[1], (tuple, list)):
                fullargspec = tf_inspect.getfullargspec(host_call[0])
                fn_args = function_utils.fn_args(host_call[0])
                if fullargspec.varargs is None and len(host_call[1]) != len(fn_args):
                    raise RuntimeError('In TPUEstimatorSpec.{}, length of tensors {} does not match method args of the function, which takes {}.'.format(name, len(host_call[1]), len(fn_args)))

    @staticmethod
    def create_cpu_hostcall(host_calls):
        """Runs on the host_call on CPU instead of TPU when use_tpu=False."""
        _OutfeedHostCall.validate(host_calls)
        ret = {}
        for name, host_call in host_calls.items():
            host_fn, tensors = host_call
            if isinstance(tensors, (tuple, list)):
                ret[name] = host_fn(*tensors)
            else:
                try:
                    ret[name] = host_fn(**tensors)
                except TypeError as e:
                    tf.compat.v1.logging.warn("Exception while calling %s: %s. It is likely the tensors (%s[1]) do not match the function's arguments", name, e, name)
                    raise
        return ret

    def record(self, host_calls):
        """Records the host_call structure."""
        for name, host_call in host_calls.items():
            host_fn, tensor_list_or_dict = host_call
            self._names.append(name)
            self._host_fns[name] = host_fn
            if isinstance(tensor_list_or_dict, dict):
                for key, tensor in six.iteritems(tensor_list_or_dict):
                    self._tensor_keys[name].append(key)
                    self._tensors[name].append(tensor)
                    self._tensor_dtypes[name].append(tensor.dtype)
                    self._tensor_shapes[name].append(tensor.shape)
            else:
                self._tensor_keys[name] = None
                for tensor in tensor_list_or_dict:
                    self._tensors[name].append(tensor)
                    self._tensor_dtypes[name].append(tensor.dtype)
                    self._tensor_shapes[name].append(tensor.shape)

    def create_enqueue_op(self, step=None):
        """Create the op to enqueue the recorded host_calls.

    Returns:
      A list of enqueue ops, which is empty if there are no host calls.
    """
        if not self._names:
            return []
        tensors = []
        for name in self._names:
            tensors.extend(self._tensors[name])
        if self._outfeed_every_n_steps > 1 and step is None:
            raise ValueError('If outfeed is requested every n steps, you must pass a tensor whose value is the step number within the current training loop.')
        with tf.compat.v1.device(tf.compat.v1.tpu.core(0)):
            if self._outfeed_every_n_steps == 1:
                return [tpu_ops.outfeed_enqueue_tuple(tensors)]
            else:
                return [tf.compat.v1.cond(tf.math.equal(tf.math.floormod(step, self._outfeed_every_n_steps), 0), lambda: tpu_ops.outfeed_enqueue_tuple(tensors), lambda: tf.no_op())]

    def create_tpu_hostcall(self):
        """Sends the tensors through outfeed and runs the host_fn on CPU.

    The tensors are concatenated along dimension 0 to form a global tensor
    across all shards. The concatenated function is passed to the host_fn and
    executed on the first host.

    Returns:
      A dictionary mapping name to the return type of the host_call by that
      name.

    Raises:
      RuntimeError: If outfeed tensor is scalar.
    """
        if not self._names:
            return {}
        ret = {}
        dequeue_ops = []
        tensor_dtypes = []
        tensor_shapes = []
        for name in self._names:
            for _ in self._tensors[name]:
                dequeue_ops.append([])
            for dtype in self._tensor_dtypes[name]:
                tensor_dtypes.append(dtype)
            for shape in self._tensor_shapes[name]:
                tensor_shapes.append(shape)
        for i in xrange(self._ctx.num_replicas):
            host_device, ordinal_id = self._ctx.device_for_replica(i)
            with tf.compat.v1.device(host_device):
                outfeed_tensors = tpu_ops.outfeed_dequeue_tuple(dtypes=tensor_dtypes, shapes=tensor_shapes, device_ordinal=ordinal_id)
                for j, item in enumerate(outfeed_tensors):
                    dequeue_ops[j].append(item)
        flat_dequeue_ops = []
        for l in dequeue_ops:
            flat_dequeue_ops.extend(l)
        dequeue_ops_by_name = {}
        pos = 0
        for name in self._names:
            dequeue_ops_by_name[name] = dequeue_ops[pos:pos + len(self._tensors[name])]
            pos += len(self._tensors[name])

        def _call_host_fn(fn, *args, **kw):
            context = CatchInvalidHostcallFunctions()
            context.Enter()
            result = fn(*args, **kw)
            context.Exit()
            context.ExitResult(result)
            return result
        with tf.compat.v1.device(self._ctx.tpu_host_placement_function(replica_id=0)):
            for name in self._names:
                dequeue_ops = dequeue_ops_by_name[name]
                for i, item in enumerate(dequeue_ops):
                    if self._ctx.config.tpu_config.per_host_input_for_training is tpu_config.InputPipelineConfig.BROADCAST:
                        with tf.control_dependencies(dequeue_ops[i]):
                            dequeue_ops[i] = tf.identity(dequeue_ops[i][0])
                    else:
                        if dequeue_ops[i][0].shape.ndims == 0:
                            raise RuntimeError('All tensors outfed from TPU should preserve batch size dimension, but got scalar {}'.format(dequeue_ops[i][0]))
                        dequeue_ops[i] = tf.concat(dequeue_ops[i], axis=0)
                if self._tensor_keys[name] is not None:
                    dequeue_ops = dict(zip(self._tensor_keys[name], dequeue_ops))
                    try:
                        ret[name] = _call_host_fn(self._host_fns[name], **dequeue_ops)
                    except TypeError as e:
                        tf.compat.v1.logging.warn("Exception while calling %s: %s. It is likely the tensors (%s[1]) do not match the function's arguments", name, e, name)
                        raise
                else:
                    ret[name] = _call_host_fn(self._host_fns[name], *dequeue_ops)
        ret['__force_dequeue'] = tf.group(*flat_dequeue_ops)
        return ret