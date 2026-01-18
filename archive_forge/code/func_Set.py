import atexit
import os
import sys
import tempfile
from absl import app
from absl.testing.absltest import *
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def Set(self, parent, child_name, new_child):
    """In parent, replace child_name's old definition with new_child.

    The parent could be a module when the child is a function at
    module scope.  Or the parent could be a class when a class' method
    is being replaced.  The named child is set to new_child, while the
    prior definition is saved away for later, when UnsetAll() is
    called.

    This method supports the case where child_name is a staticmethod or a
    classmethod of parent.

    Args:
      parent: The context in which the attribute child_name is to be changed.
      child_name: The name of the attribute to change.
      new_child: The new value of the attribute.
    """
    old_child = getattr(parent, child_name)
    old_attribute = parent.__dict__.get(child_name)
    if old_attribute is not None and isinstance(old_attribute, staticmethod):
        old_child = staticmethod(old_child)
    self.cache.append((parent, old_child, child_name))
    setattr(parent, child_name, new_child)