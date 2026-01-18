import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
class Defun(object):
    """Obsolete. Slated for deletion. Please use tf.function instead.

  Known feature gaps while migrating to tf.function (could be outdated):
  - tf.function doesn’t support Send/Recv capability since it doesn’t share
    rendezvous with the main graph but always creates a new one.
  - tf.function doesn’t support custom gradient function directly, instead you
    need to define the function inside a tf.custom_gradient wrapper together
    with the gradient function.
  - Unlike Defun, Keras layers used inside a tf.function need to be created only
    once to avoid variable recreation.
  - Defun respects the device assignments and applies them to the function body
    but tf.function needs it to be done manually.
  - Defun might prune out unused ops automatically but tf.function doesn't.

  Limitations of Defun:
  - Original source locations are not preserved so errors do not include
    full/valid stack traces.
  - Only supports linear sequence of arguments and return values, putting the
    burden on the caller to pack/unpack everything across a Defun boundary into
    tuples (as opposed to passing list and dict-like structures directly).
  - Does not support overloading or late-bound specializations.
  - Has its own way for defining gradient overrides which does not follow
    current conventions.
  - Cannot support imperative control flow or automatic control dependencies.
  - Does not reflect statefulness in the graph and has a calling convention that
    differs from how more modern tools interact.
  - Is only compatible with graph building mode.

  Decorator used to define TensorFlow functions.

  Use this decorator to make a Python function usable directly as a TensorFlow
  function.

  The decorated function must add ops to the default graph and return zero or
  more `Tensor` objects.  Call the decorator with named arguments, one for each
  argument of the function to decorate, with the expected type of the argument
  as value.

  For example if the function to decorate accepts two `tf.float32` arguments
  named `x` and `y`, call the decorator with:

      @Defun(tf.float32, tf.float32)
      def foo(x, y):
        ...

  When you call the decorated function, it adds the `call` ops to the
  default graph. In addition, it adds the definition of the function into the
  default graph. Because the addition of the function into the graph
  is deferred, the decorator can be used anywhere in the program.

  Any variables created inside of the function are hoisted into the outer graph.
  Note that the variables are created in the variable scope that was active
  during the first call to the function. Subsequent function calls will refer to
  the same set of variables.

  Definitions of functions in a graph are frozen as soon as the graph is used to
  create a session. However, new functions and new calls to existing functions
  may be added to the graph, with the new functions themselves becoming
  immediately frozen.

  Example, but also see the [How To on functions](link_needed).

  ```python
  # Defining the function.
  @tf.Defun(tf.float32, tf.float32)
  def MyFunc(x, y):
    return x + y, x - y

  # Building the graph.
  a = tf.constant([1.0])
  b = tf.constant([2.0])
  c, d = MyFunc(a, b, name='mycall')
  ```
  """

    def __init__(self, *input_types, **kwargs):
        """Create a `Defun` decorator.

    Args:
      *input_types: A list of `tf.DType`
      **kwargs: Optional keyword arguments, including
         func_name - (optional).  A python string, the name to use to
           declare this `Function` in the graph.

         grad_func - (optional).  A function implementing the gradient
           of the function-to-register.  This is must be a
           `_DefinedFunction` object. The gradient
           function must satisfy the criterion defined in
           function.proto:GradientDef.

         python_grad_func - (optional).  A function implementing the
           gradient of the function python-side. This function must
           take the current op and the gradients w.r.t. its outputs,
           and return the gradients w.r.t. the inputs. That is it must
           implement the interface expected by `tf.RegisterGradient`).
           This will be called by tf.gradients to add the gradient ops
           to the graph. At most one of grad_func and python_grad_func
           can be specified.

         out_names = (optional). A list of strings, one per output
           tensor.

         shape_func - (optional). A function taking the op and returning a list
           of static shapes to set for the function's outputs.
    """
        self._input_types = input_types
        self._func_name = kwargs.pop('func_name', None)
        self._grad_func = kwargs.pop('grad_func', None)
        self._python_grad_func = kwargs.pop('python_grad_func', None)
        self._out_names = kwargs.pop('out_names', None)
        self._extra_kwargs = kwargs

    def __call__(self, func):
        if not callable(func):
            raise ValueError(f'Function {func} must be a callable.')
        argspec = tf_inspect.getargspec(func)
        if argspec.keywords or argspec.defaults:
            raise ValueError(f'Functions with argument defaults or keywords arguments are not supported. {func} has defaults {argspec.defaults} and keywords {argspec.keywords}.')
        min_args = len(argspec.args)
        max_args = min_args
        if argspec.varargs:
            max_args = 1000000
        argnames = argspec.args
        if tf_inspect.ismethod(func):
            min_args -= 1
            argnames = argnames[1:]
        if self._input_types:
            num = len(self._input_types)
            if num < min_args or num > max_args:
                raise ValueError(f'The number of tf.function input types is not compatible with the allowed arguments of {func}. The tf.function have {num} input types, while the python function allows minimum {min_args} and maximum {max_args} arguments.')
            return _DefinedFunction(func, argnames, self._input_types, self._func_name, self._grad_func, self._python_grad_func, out_names=self._out_names, **self._extra_kwargs)
        if min_args == 0 and max_args == 0:
            return _DefinedFunction(func, [], [], self._func_name, self._grad_func, self._python_grad_func, out_names=self._out_names, **self._extra_kwargs)
        return _OverloadedFunction(func, argnames, self._func_name, self._grad_func, self._python_grad_func, out_names=self._out_names, **self._extra_kwargs)