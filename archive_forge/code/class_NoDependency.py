import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class NoDependency:
    """Allows attribute assignment to `Trackable` objects with no dependency.

  Example usage:
  ```python
  obj = Trackable()
  obj.has_dependency = tf.Variable(0., name="dep")
  obj.no_dependency = NoDependency(tf.Variable(1., name="nodep"))
  assert obj.no_dependency.name == "nodep:0"
  ```

  `obj` in this example has a dependency on the variable "dep", and both
  attributes contain un-wrapped `Variable` objects.

  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  """
    __slots__ = ['value']

    def __init__(self, value):
        self.value = value