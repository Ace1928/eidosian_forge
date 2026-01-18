import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def _getargspec(target):
    """A python3 version of getargspec.

    Calls `getfullargspec` and assigns args, varargs,
    varkw, and defaults to a python 2/3 compatible `ArgSpec`.

    The parameter name 'varkw' is changed to 'keywords' to fit the
    `ArgSpec` struct.

    Args:
      target: the target object to inspect.

    Returns:
      An ArgSpec with args, varargs, keywords, and defaults parameters
      from FullArgSpec.
    """
    fullargspecs = getfullargspec(target)
    if hasattr(_inspect, 'ArgSpec'):
        argspecs = ArgSpec(args=fullargspecs.args, varargs=fullargspecs.varargs, keywords=fullargspecs.varkw, defaults=fullargspecs.defaults)
    else:
        argspecs = FullArgSpec(args=fullargspecs.args, varargs=fullargspecs.varargs, varkw=fullargspecs.varkw, defaults=fullargspecs.defaults, kwonlyargs=[], kwonlydefaults=None, annotations={})
    return argspecs