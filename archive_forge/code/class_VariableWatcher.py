import contextlib
from tensorflow.python import pywrap_tfe
class VariableWatcher(object):
    """A scope that tracks all trainable variable accesses within it.

  This explicitly ignores variables that are not marked as trainable.

  Sample usage:

  var = tf.Variable(0.0)
  with VariableWatcher() as variable_watcher:
    var.assign_add(1.0)

  assert variable_watcher.watched_variables == [var]
  """
    __slots__ = ['_variable_watcher']

    def __init__(self):
        self._variable_watcher = None

    def __enter__(self):
        self._variable_watcher = pywrap_tfe.TFE_Py_VariableWatcherNew()
        return self

    def __exit__(self, typ, value, traceback):
        pywrap_tfe.TFE_Py_VariableWatcherRemove(self._variable_watcher)

    def watched_variables(self):
        """Returns a tuple of variables accessed under this scope."""
        return pywrap_tfe.TFE_Py_VariableWatcherWatchedVariables(self._variable_watcher)