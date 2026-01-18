import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
def _get_typename(obj):
    """Return human readable pretty type name string."""
    objtype = type(obj)
    name = objtype.__name__
    module = getattr(objtype, '__module__', None)
    if module:
        return '{}.{}'.format(module, name)
    else:
        return name