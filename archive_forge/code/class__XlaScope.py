import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
class _XlaScope(object):
    """Keeps track of previous XLA scope calls, and depth of current call."""

    def __init__(self, count, depth):
        self.count = count
        self.depth = depth