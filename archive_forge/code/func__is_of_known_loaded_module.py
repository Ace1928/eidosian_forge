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
def _is_of_known_loaded_module(f, module_name):
    mod = sys.modules.get(module_name, None)
    if mod is None:
        return False
    if any((v is not None for v in mod.__dict__.values() if f is v)):
        return True
    return False