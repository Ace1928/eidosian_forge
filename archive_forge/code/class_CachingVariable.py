import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
class CachingVariable(resource_variable_ops.BaseResourceVariable, core.Tensor):
    """A wrapper around a variable that caches read value locally."""

    def __init__(self, v):
        self._v = v
        self._cache = None
        self._current_new_cache_scope_count = 0

    def get(self):
        return self._v

    def __getattr__(self, name):
        return getattr(self._v, name)

    def read_value(self):
        if distribute_utils.caching_scope_local.in_caching_scope():
            return self.cached_read_value()
        return self._v.read_value()

    def sparse_read(self, indices, name=None):
        return self._v.sparse_read(indices, name=name)

    def cached_read_value(self):
        if distribute_utils.caching_scope_local.new_cache_scope_count > self._current_new_cache_scope_count:
            self._current_new_cache_scope_count += 1
            self._cache = None
        with ops.device('CPU:0'):
            if self._cache is not None:
                return self._cache
            else:
                self._cache = array_ops.identity(self._v)
                return self._cache

    def assign_sub(self, *args, **kwargs):
        return self._v.assign_sub(*args, **kwargs)

    def assign_add(self, *args, **kwargs):
        return self._v.assign_add(*args, **kwargs)

    def assign(self, *args, **kwargs):
        return self._v.assign(*args, **kwargs)

    @property
    def initializer(self):
        return self._v.initializer

    def initialized_value(self):
        return self._v.initialized_value()

    @property
    def initial_value(self):
        return self._v.initial_value

    @property
    def op(self):
        return self._v.op

    def value(self):
        if distribute_utils.caching_scope_local.in_caching_scope():
            return self.cached_read_value()
        return self._v.value()

    def eval(self, session=None):
        return self._v.eval(session)

    @property
    def graph(self):
        return self._v.graph

    @property
    def device(self):
        return self._v.device

    @property
    def shape(self):
        return self._v.shape

    @property
    def synchronization(self):
        return self._v.synchronization

    @property
    def name(self):
        return self._v.name

    @property
    def trainable(self):
        return self._v.trainable

    @property
    def dtype(self):
        return self._v.dtype

    @property
    def constraint(self):
        return self._v.constraint

    def __array__(self, dtype=None):
        return np.asarray(self.numpy(), dtype=dtype)

    def __complex__(self):
        return complex(self.value().numpy())

    def __int__(self):
        return int(self.value().numpy())

    def __float__(self):
        return float(self.value().numpy())

    def numpy(self):
        if context.executing_eagerly():
            return self.read_value().numpy()
        else:
            raise NotImplementedError('numpy() is only available when eager execution is enabled.')

    def __str__(self):
        return str(self._v)

    def __repr__(self):
        return repr(self._v)

    def _should_act_as_resource_variable(self):
        """Pass resource_variable_ops.is_resource_variable check."""
        pass

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        if distribute_utils.caching_scope_local.in_caching_scope():
            return self.cached_read_value()
        return self._v._dense_var_to_tensor(dtype=dtype, name=name, as_ref=False)

    @classmethod
    def _overload_overloadable_operators(cls):
        """Register overloads for all operators."""
        for operator in tensor.Tensor.OVERLOADABLE_OPERATORS:
            if operator == '__eq__' or operator == '__ne__':
                continue
            cls._tensor_overload_operator(operator)

    @classmethod
    def _tensor_overload_operator(cls, operator):
        """Delegate an operator overload to `tensor.Tensor`."""
        tensor_operator = getattr(tensor.Tensor, operator)

        def _operator(v, *args, **kwargs):
            return tensor_operator(v.value(), *args, **kwargs)
        setattr(cls, operator, _operator)

    def _gather_saveables_for_checkpoint(self):
        return {trackable.VARIABLE_VALUE_KEY: self._v}

    def _export_to_saved_model_graph(self, object_map, tensor_map, options, **kwargs):
        """For implementing `Trackable`."""
        resource_list = self._v._export_to_saved_model_graph(object_map, tensor_map, options, **kwargs)
        object_map[self] = object_map[self._v]
        return resource_list