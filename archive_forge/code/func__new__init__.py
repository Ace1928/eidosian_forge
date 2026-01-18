import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
def _new__init__(self, wrapped_value, tf_should_use_helper):
    self._tf_should_use_helper = tf_should_use_helper
    self._tf_should_use_wrapped_value = wrapped_value