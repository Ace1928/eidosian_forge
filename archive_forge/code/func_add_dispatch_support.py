import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.dispatch.add_dispatch_support', v1=[])
def add_dispatch_support(target=None, iterable_parameters=None):
    """Decorator that adds a dispatch handling wrapper to a TensorFlow Python API.

  This wrapper adds the decorated function as an API that can be overridden
  using the `@dispatch_for_api` decorator.  In the following example, we first
  define a new API (`double`) that supports dispatch, then define a custom type
  (`MaskedTensor`) and finally use `dispatch_for_api` to override the default
  implementation of `double` when called with `MaskedTensor` values:

  >>> @add_dispatch_support
  ... def double(x):
  ...   return x * 2
  >>> class MaskedTensor(tf.experimental.ExtensionType):
  ...   values: tf.Tensor
  ...   mask: tf.Tensor
  >>> @dispatch_for_api(double, {'x': MaskedTensor})
  ... def masked_double(x):
  ...   return MaskedTensor(x.values * 2, y.mask)

  The optional `iterable_parameter` argument can be used to mark parameters that
  can take arbitrary iterable values (such as generator expressions).  These
  need to be handled specially during dispatch, since just iterating over an
  iterable uses up its values.  In the following example, we define a new API
  whose second argument can be an iterable value; and then override the default
  implementatio of that API when the iterable contains MaskedTensors:

  >>> @add_dispatch_support(iterable_parameters=['ys'])
  ... def add_tensor_to_list_of_tensors(x, ys):
  ...   return [x + y for y in ys]
  >>> @dispatch_for_api(add_tensor_to_list_of_tensors,
  ...               {'ys': typing.List[MaskedTensor]})
  ... def masked_add_tensor_to_list_of_tensors(x, ys):
  ...   return [MaskedTensor(x+y.values, y.mask) for y in ys]

  (Note: the only TensorFlow API that currently supports iterables is `add_n`.)

  Args:
    target: The TensorFlow API that should support dispatch.
    iterable_parameters: Optional list of parameter names that may be called
      with iterables (such as the `inputs` parameter for `tf.add_n`).

  Returns:
    A decorator.
  """
    if not (iterable_parameters is None or (isinstance(iterable_parameters, (list, tuple)) and all((isinstance(p, str) for p in iterable_parameters)))):
        raise TypeError('iterable_parameters should be a list or tuple of string.')

    def decorator(dispatch_target):
        if iterable_parameters is None:
            iterable_params = None
        else:
            arg_names = tf_inspect.getargspec(dispatch_target).args
            iterable_params = [(name, arg_names.index(name)) for name in iterable_parameters]

        @traceback_utils.filter_traceback
        def op_dispatch_handler(*args, **kwargs):
            """Call `dispatch_target`, peforming dispatch when appropriate."""
            if api_dispatcher is not None:
                if iterable_params is not None:
                    args, kwargs = replace_iterable_params(args, kwargs, iterable_params)
                result = api_dispatcher.Dispatch(args, kwargs)
                if result is not NotImplemented:
                    return result
            try:
                return dispatch_target(*args, **kwargs)
            except (TypeError, ValueError):
                result = dispatch(op_dispatch_handler, args, kwargs)
                if result is not OpDispatcher.NOT_SUPPORTED:
                    return result
                else:
                    raise
        add_fallback_dispatch_list(op_dispatch_handler)
        op_dispatch_handler = tf_decorator.make_decorator(dispatch_target, op_dispatch_handler)
        add_type_based_api_dispatcher(op_dispatch_handler)
        api_dispatcher = getattr(op_dispatch_handler, TYPE_BASED_DISPATCH_ATTR, None)
        return op_dispatch_handler
    if target is None:
        return decorator
    else:
        return decorator(target)