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
def _add_name_scope_wrapper(func, api_signature):
    """Wraps `func` to expect a "name" arg, and use it to call `ops.name_scope`.

  If `func` already expects a "name" arg, or if `api_signature` does not
  expect a "name" arg, then returns `func` as-is.

  Args:
    func: The function to wrap.  Signature must match `api_signature` (except
      the "name" parameter may be missing.
    api_signature: The signature of the original API (used to find the index for
      the "name" parameter).

  Returns:
    The wrapped function (or the original function if no wrapping is needed).
  """
    if 'name' not in api_signature.parameters:
        return func
    func_signature = tf_inspect.signature(func)
    func_argspec = tf_inspect.getargspec(func)
    if 'name' in func_signature.parameters or func_argspec.keywords is not None:
        return func
    name_index = list(api_signature.parameters).index('name')

    def wrapped_func(*args, **kwargs):
        if name_index < len(args):
            name = args[name_index]
            args = args[:name_index] + args[name_index + 1:]
        else:
            name = kwargs.pop('name', None)
        if name is None:
            return func(*args, **kwargs)
        else:
            with ops.name_scope(name):
                return func(*args, **kwargs)
    wrapped_func = tf_decorator.make_decorator(func, wrapped_func)
    wrapped_func.__signature__ = func_signature.replace(parameters=list(func_signature.parameters.values()) + [api_signature.parameters['name']])
    del wrapped_func._tf_decorator
    return wrapped_func