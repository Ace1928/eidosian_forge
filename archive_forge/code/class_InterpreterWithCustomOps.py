import ctypes
import enum
import os
import platform
import sys
import numpy as np
class InterpreterWithCustomOps(Interpreter):
    """Interpreter interface for TensorFlow Lite Models that accepts custom ops.

  The interface provided by this class is experimental and therefore not exposed
  as part of the public API.

  Wraps the tf.lite.Interpreter class and adds the ability to load custom ops
  by providing the names of functions that take a pointer to a BuiltinOpResolver
  and add a custom op.
  """

    def __init__(self, custom_op_registerers=None, **kwargs):
        """Constructor.

    Args:
      custom_op_registerers: List of str (symbol names) or functions that take a
        pointer to a MutableOpResolver and register a custom op. When passing
        functions, use a pybind function that takes a uintptr_t that can be
        recast as a pointer to a MutableOpResolver.
      **kwargs: Additional arguments passed to Interpreter.

    Raises:
      ValueError: If the interpreter was unable to create.
    """
        self._custom_op_registerers = custom_op_registerers or []
        super(InterpreterWithCustomOps, self).__init__(**kwargs)