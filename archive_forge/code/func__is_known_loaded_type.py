import functools
import inspect
import sys
import unittest
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.util import tf_inspect
def _is_known_loaded_type(f, module_name, entity_name):
    """Tests whether the function or method is an instance of a known type."""
    if module_name not in sys.modules or not hasattr(sys.modules[module_name], entity_name):
        return False
    type_entity = getattr(sys.modules[module_name], entity_name)
    if isinstance(f, type_entity):
        return True
    if inspect.ismethod(f):
        if isinstance(f.__func__, type_entity):
            return True
    return False