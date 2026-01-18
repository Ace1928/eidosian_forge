import abc
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _consumers(self):
    """Returns a list of `Operation`s that consume this `CompositeTensor`.

    Returns:
      A list of `Operation`s.

    Raises:
      RuntimeError: If this method is called while executing eagerly.
    """
    consumers = nest.flatten([component.consumers() for component in nest.flatten(self, expand_composites=True) if getattr(component, 'graph', None) is not None])
    return list(set(consumers))