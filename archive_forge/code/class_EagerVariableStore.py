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
class EagerVariableStore:
    """Wrapper allowing functional layers to be used with eager execution.

  When eager execution is enabled Variables get deleted when they go out of
  scope, and are not stored in global collections by default. A lot of code
  (mostly the functional layers in tf.layers) assumes that variables are kept in
  a global list.

  EagerVariableStore can be used in conjunction with this code to make it
  eager-friendly. For example, to create a dense layer, use:

  ```
    container = tfe.EagerVariableStore()
    for input in dataset_iterator:
      with container.as_default():
        x = tf.compat.v1.layers.dense(input, name="l1")
    print(container.variables)  # Should print the variables used in the layer.
  ```
  """

    def __init__(self, store=None):
        if store is not None:
            if not store._store_eager_variables:
                raise ValueError('Cannot construct EagerVariableStore from a VariableStore object that does not hold eager variables.')
            self._store = store
        else:
            self._store = _VariableStore()
        self._store._store_eager_variables = True

    def as_default(self):
        return with_variable_store(self._store)

    def variables(self):
        return sorted(self._store._vars.values(), key=lambda x: x.name)

    def trainable_variables(self):
        return sorted([x for x in self._store._vars.values() if x.trainable], key=lambda x: x.name)

    def non_trainable_variables(self):
        return sorted([x for x in self._store._vars.values() if not x.trainable], key=lambda x: x.name)

    def copy(self):
        """Copy this variable store and all of its contents.

    Variables contained in this store will be copied over to the new variable
    store, meaning that they can be modified without affecting the variables in
    this store.

    Returns:
      A new EagerVariableStore instance containing copied variables.
    """
        new_store = EagerVariableStore()
        for key, var in self._store._vars.items():
            try:
                index = var.name.index(':')
            except ValueError:
                stripped_var_name = var.name
            else:
                stripped_var_name = var.name[:index]
            new_var = resource_variable_ops.ResourceVariable(var.read_value(), name=stripped_var_name, trainable=var.trainable)
            new_store._store._vars[key] = new_var
        return new_store