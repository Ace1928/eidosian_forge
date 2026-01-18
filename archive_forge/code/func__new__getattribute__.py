import copy
import sys
import textwrap
import traceback
import types
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_decorator
def _new__getattribute__(self, key):
    if key not in ('_tf_should_use_helper', '_tf_should_use_wrapped_value'):
        object.__getattribute__(self, '_tf_should_use_helper').sate()
    if key in ('_tf_should_use_wrapped_value', '_tf_should_use_helper', 'mark_used', '__setattr__'):
        return object.__getattribute__(self, key)
    return getattr(object.__getattribute__(self, '_tf_should_use_wrapped_value'), key)