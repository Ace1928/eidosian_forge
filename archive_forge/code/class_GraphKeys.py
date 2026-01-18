import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['GraphKeys'])
class GraphKeys(object):
    """Standard names to use for graph collections.

  The standard library uses various well-known names to collect and
  retrieve values associated with a graph. For example, the
  `tf.Optimizer` subclasses default to optimizing the variables
  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  specified, but it is also possible to pass an explicit list of
  variables.

  The following standard keys are defined:

  * `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
    across distributed environment (model variables are subset of these). See
    `tf.compat.v1.global_variables`
    for more details.
    Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
    and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
  * `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
    machine. Usually used for temporarily variables, like counters.
  * `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
    model for inference (feed forward).
  * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
    be trained by an optimizer. See
    `tf.compat.v1.trainable_variables`
    for more details.
  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    graph. See
    `tf.compat.v1.summary.merge_all`
    for more details.
  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    produce input for a computation. See
    `tf.compat.v1.train.start_queue_runners`
    for more details.
  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    keep moving averages.  See
    `tf.compat.v1.moving_average_variables`
    for more details.
  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    construction.

  The following standard keys are _defined_, but their collections are **not**
  automatically populated as many of the others are:

  * `WEIGHTS`
  * `BIASES`
  * `ACTIVATIONS`
  """
    GLOBAL_VARIABLES = 'variables'
    LOCAL_VARIABLES = 'local_variables'
    METRIC_VARIABLES = 'metric_variables'
    MODEL_VARIABLES = 'model_variables'
    TRAINABLE_VARIABLES = 'trainable_variables'
    SUMMARIES = 'summaries'
    QUEUE_RUNNERS = 'queue_runners'
    TABLE_INITIALIZERS = 'table_initializer'
    ASSET_FILEPATHS = 'asset_filepaths'
    MOVING_AVERAGE_VARIABLES = 'moving_average_variables'
    REGULARIZATION_LOSSES = 'regularization_losses'
    CONCATENATED_VARIABLES = 'concatenated_variables'
    SAVERS = 'savers'
    WEIGHTS = 'weights'
    BIASES = 'biases'
    ACTIVATIONS = 'activations'
    UPDATE_OPS = 'update_ops'
    LOSSES = 'losses'
    SAVEABLE_OBJECTS = 'saveable_objects'
    RESOURCES = 'resources'
    LOCAL_RESOURCES = 'local_resources'
    TRAINABLE_RESOURCE_VARIABLES = 'trainable_resource_variables'
    INIT_OP = 'init_op'
    LOCAL_INIT_OP = 'local_init_op'
    READY_OP = 'ready_op'
    READY_FOR_LOCAL_INIT_OP = 'ready_for_local_init_op'
    SUMMARY_OP = 'summary_op'
    GLOBAL_STEP = 'global_step'
    EVAL_STEP = 'eval_step'
    TRAIN_OP = 'train_op'
    COND_CONTEXT = 'cond_context'
    WHILE_CONTEXT = 'while_context'
    _SUMMARY_COLLECTION = '_SUMMARY_V2'
    _VARIABLE_COLLECTIONS = [GLOBAL_VARIABLES, LOCAL_VARIABLES, METRIC_VARIABLES, MODEL_VARIABLES, TRAINABLE_VARIABLES, MOVING_AVERAGE_VARIABLES, CONCATENATED_VARIABLES, TRAINABLE_RESOURCE_VARIABLES]
    _STREAMING_MODEL_PORTS = 'streaming_model_ports'

    @decorator_utils.classproperty
    @deprecation.deprecated(None, 'Use `tf.GraphKeys.GLOBAL_VARIABLES` instead.')
    def VARIABLES(cls):
        return cls.GLOBAL_VARIABLES