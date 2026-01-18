import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _lazy_init_variable_creator(next_creator, **kwargs):
    if getattr(_DISABLE_LAZY_VARIABLE_INIT, 'disabled', False):
        return next_creator(**kwargs)
    else:
        return LazyInitVariable(**kwargs)