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
class _DefaultDistributionContext(object):
    """Context manager setting the default `tf.distribute.Strategy`."""
    __slots__ = ['_var_creator_scope', '_strategy', '_nested_count']

    def __init__(self, strategy):

        def creator(next_creator, **kwargs):
            _require_strategy_scope_strategy(strategy)
            return next_creator(**kwargs)
        self._var_creator_scope = variable_scope.variable_creator_scope(creator)
        self._strategy = strategy
        self._nested_count = 0

    def __enter__(self):
        if has_strategy():
            raise RuntimeError('Must not nest tf.distribute.Strategy scopes.')
        if self._nested_count == 0:
            self._var_creator_scope.__enter__()
        self._nested_count += 1
        return self._strategy

    def __exit__(self, exception_type, exception_value, traceback):
        self._nested_count -= 1
        if self._nested_count == 0:
            try:
                self._var_creator_scope.__exit__(exception_type, exception_value, traceback)
            except RuntimeError as e:
                six.raise_from(RuntimeError('Variable creator scope nesting error: move call to tf.distribute.set_strategy() out of `with` scope.'), e)