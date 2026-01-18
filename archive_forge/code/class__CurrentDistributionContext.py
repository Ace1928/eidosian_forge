import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class _CurrentDistributionContext(object):
    """Context manager setting the current `tf.distribute.Strategy`.

  Also: overrides the variable creator and optionally the current device.
  """

    def __init__(self, strategy, var_creator_scope, var_scope=None, resource_creator_scope=None, default_device_scope=None):
        self._context = _CrossReplicaThreadMode(strategy)
        self._var_creator_scope = var_creator_scope
        self._var_scope = var_scope
        self._resource_creator_scope = resource_creator_scope
        if default_device_scope:
            self._device_scope = default_device_scope
        else:
            self._device_scope = None
        self._same_scope_again_count = 0

    def __enter__(self):
        if has_strategy():
            _require_cross_replica_or_default_context_extended(self._context.strategy.extended)
            self._same_scope_again_count += 1
        else:
            _push_per_thread_mode(self._context)
            if self._var_scope:
                self._var_scope.__enter__()
            self._var_creator_scope.__enter__()
            if self._resource_creator_scope:
                nest.map_structure(lambda scope: scope.__enter__(), self._resource_creator_scope)
            if self._device_scope:
                self._device_scope.__enter__()
        return self._context.strategy

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(self._context.strategy.extended, '_lazy_variable_tracker'):
            self._context.strategy.extended._lazy_variable_tracker.initialize_all()
        if self._same_scope_again_count > 0:
            self._same_scope_again_count -= 1
            return
        if self._device_scope:
            try:
                self._device_scope.__exit__(exception_type, exception_value, traceback)
            except RuntimeError as e:
                six.raise_from(RuntimeError('Device scope nesting error: move call to tf.distribute.set_strategy() out of `with` scope.'), e)
        try:
            self._var_creator_scope.__exit__(exception_type, exception_value, traceback)
        except RuntimeError as e:
            six.raise_from(RuntimeError('Variable creator scope nesting error: move call to tf.distribute.set_strategy() out of `with` scope.'), e)
        if self._resource_creator_scope:
            try:
                if isinstance(self._resource_creator_scope, list):
                    reversed_resource_creator_scope = self._resource_creator_scope[::-1]
                    nest.map_structure(lambda scope: scope.__exit__(exception_type, exception_value, traceback), reversed_resource_creator_scope)
                else:
                    self._resource_creator_scope.__exit__(exception_type, exception_value, traceback)
            except RuntimeError as e:
                six.raise_from(RuntimeError('Resource creator scope nesting error: move call to tf.distribute.set_strategy() out of `with` scope.'), e)
        if self._var_scope:
            try:
                self._var_scope.__exit__(exception_type, exception_value, traceback)
            except RuntimeError as e:
                six.raise_from(RuntimeError('Variable scope nesting error: move call to tf.distribute.set_strategy() out of `with` scope.'), e)
        _pop_per_thread_mode()