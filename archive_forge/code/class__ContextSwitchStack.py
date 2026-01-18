import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
class _ContextSwitchStack(threading.local):
    """A thread-local stack of context switches."""

    def __init__(self, eager):
        super().__init__()
        self.stack = []
        if eager:
            self.push(is_building_function=False, enter_context_fn=eager_mode, device_stack=None)

    def push(self, is_building_function, enter_context_fn, device_stack):
        """Push metadata about a context switch onto the stack.

    A context switch can take any one of the two forms: installing a graph as
    the default graph, or entering the eager context. For each context switch,
    we record whether or not the entered context is building a function.

    Args:
      is_building_function: (bool.) Whether the context is building a function.
      enter_context_fn: (function.) A callable that executes the context switch.
        For example, `graph.as_default` or `eager_mode`.
      device_stack: If applicable, the device function stack for this graph.
        When breaking out of graphs in init_scope, the innermost nonempty device
        stack is used. Eager contexts put `None` here and the value is never
        used.
    """
        self.stack.append(ContextSwitch(is_building_function, enter_context_fn, device_stack))

    def pop(self):
        """Pop the stack."""
        self.stack.pop()