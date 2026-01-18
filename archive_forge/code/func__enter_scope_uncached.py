import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _enter_scope_uncached(self):
    """Enters the context manager when there is no cached scope yet.

    Returns:
      The entered variable scope.

    Raises:
      TypeError: A wrong type is passed as `scope` at __init__().
      ValueError: `reuse` is incorrectly set at __init__().
    """
    if self._auxiliary_name_scope:
        current_name_scope = None
    else:
        name_scope = ops.get_name_scope()
        if name_scope:
            name_scope += '/'
            current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
        else:
            current_name_scope = ops.name_scope(name_scope, skip_on_eager=False)
    if self._name_or_scope is not None:
        if not isinstance(self._name_or_scope, (VariableScope, str)):
            raise TypeError('VariableScope: name_or_scope must be a string or VariableScope.')
        if isinstance(self._name_or_scope, str):
            name_scope = self._name_or_scope
        else:
            name_scope = self._name_or_scope.name.split('/')[-1]
        if name_scope or current_name_scope:
            current_name_scope = current_name_scope or ops.name_scope(name_scope, skip_on_eager=False)
            try:
                current_name_scope_name = current_name_scope.__enter__()
            except:
                current_name_scope.__exit__(*sys.exc_info())
                raise
            self._current_name_scope = current_name_scope
            if isinstance(self._name_or_scope, str):
                old_name_scope = current_name_scope_name
            else:
                old_name_scope = self._name_or_scope.original_name_scope
            pure_variable_scope = _pure_variable_scope(self._name_or_scope, reuse=self._reuse, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, old_name_scope=old_name_scope, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
            try:
                entered_pure_variable_scope = pure_variable_scope.__enter__()
            except:
                pure_variable_scope.__exit__(*sys.exc_info())
                raise
            self._cached_pure_variable_scope = pure_variable_scope
            return entered_pure_variable_scope
        else:
            self._current_name_scope = None
            pure_variable_scope = _pure_variable_scope(self._name_or_scope, reuse=self._reuse, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
            try:
                entered_pure_variable_scope = pure_variable_scope.__enter__()
            except:
                pure_variable_scope.__exit__(*sys.exc_info())
                raise
            self._cached_pure_variable_scope = pure_variable_scope
            return entered_pure_variable_scope
    else:
        if self._reuse:
            raise ValueError('reuse=True cannot be used without a name_or_scope')
        current_name_scope = current_name_scope or ops.name_scope(self._default_name, skip_on_eager=False)
        try:
            current_name_scope_name = current_name_scope.__enter__()
        except:
            current_name_scope.__exit__(*sys.exc_info())
            raise
        self._current_name_scope = current_name_scope
        unique_default_name = _get_unique_variable_scope(self._default_name)
        pure_variable_scope = _pure_variable_scope(unique_default_name, initializer=self._initializer, regularizer=self._regularizer, caching_device=self._caching_device, partitioner=self._partitioner, custom_getter=self._custom_getter, old_name_scope=current_name_scope_name, dtype=self._dtype, use_resource=self._use_resource, constraint=self._constraint)
        try:
            entered_pure_variable_scope = pure_variable_scope.__enter__()
        except:
            pure_variable_scope.__exit__(*sys.exc_info())
            raise
        self._cached_pure_variable_scope = pure_variable_scope
        return entered_pure_variable_scope