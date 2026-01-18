from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
class Bind:
    """When called evaluates `d(f, args, kwargs)` but supports binding `f`.

  >>> @Bind.decorator
  ... def my_decorator(f, args, kwargs):
  ...   print("my_decorator called with", args, kwargs)
  ...   return f(*args, **kwargs)

  >>> class Foo:
  ...   @my_decorator
  ...   def bar(self, a, b, c):
  ...     return a * b * c

  >>> Foo.bar(None, 1, 2, c=3)
  my_decorator called with (None, 1, 2) {'c': 3}
  6

  >>> foo = Foo()
  >>> foo.bar(1, 2, c=3)
  my_decorator called with (1, 2) {'c': 3}
  6
  """

    @classmethod
    def decorator(cls, d):
        return lambda f: Bind(f, d)

    def __init__(self, f, d):
        self._f = f
        self._d = d

    def __get__(self, instance, owner):
        if instance is not None:
            f = self._f.__get__(instance, owner)
            return tf_decorator.make_decorator(f, Bind(f, self._d))
        else:
            return self

    def __call__(self, *a, **k):
        return self._d(self._f, a, k)