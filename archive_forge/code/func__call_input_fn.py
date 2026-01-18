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
def _call_input_fn(self, input_fn, mode, input_context=None):
    """Calls the input function.

    Args:
      input_fn: The input function.
      mode: ModeKeys
      input_context: Optional instance of `tf.distribute.InputContext`.

    Returns:
      In TPU mode, returns an input_fn to be called later in model_fn.
      Otherwise, calls the input_fn and returns either fatures or
        (features, labels).

    Raises:
      ValueError: if input_fn takes invalid arguments or does not have `params`.
    """
    input_fn_args = function_utils.fn_args(input_fn)
    config = self.config
    kwargs = {}
    if 'params' in input_fn_args:
        kwargs['params'] = self.params
    else:
        raise ValueError('input_fn ({}) does not include params argument, required by TPUEstimator to pass batch size as params["batch_size"]'.format(input_fn))
    if 'config' in input_fn_args:
        kwargs['config'] = config
    if 'mode' in input_fn_args:
        kwargs['mode'] = mode
    if 'input_context' in input_fn_args:
        kwargs['input_context'] = input_context
    self._is_input_fn_invoked = True
    with self._ctx.with_mode(mode) as ctx:
        if ctx.is_running_on_cpu() and ctx.is_input_slice_broadcast_to_all_cores():
            raise ValueError('Invalid TPUConfig `eval_training_input_configuration` value. SLICED mode only works on use_tpu = True.')
        batch_size_for_input_fn = ctx.batch_size_for_input_fn
        if batch_size_for_input_fn is not None:
            _add_item_to_params(kwargs['params'], _BATCH_SIZE_KEY, batch_size_for_input_fn)
        if ctx.is_running_on_cpu(is_export_mode=False):
            with tf.compat.v1.device('/device:CPU:0'):
                return input_fn(**kwargs)

        def _input_fn(ctx):
            _add_item_to_params(kwargs['params'], _CTX_KEY, ctx)
            return input_fn(**kwargs)
        return _input_fn