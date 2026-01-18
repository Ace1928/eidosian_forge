import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
@classmethod
def __tf_unflatten__(cls, metadata, components):
    """Create a user-defined object from (metadata, components).

    Args:
      metadata: a custom Python objet that stands for the static config for
        reconstructing a new object of the current class.
      components: a `tuple` that contains the dynamic data fields of the current
        class, for object reconstruction.

    Returns:
      The user-defined object, with the same class of the current object.

    Implementation Note:
    - This method should not invoke any TensorFlow ops.
    - This method only needs to unflatten the current level. If the object has
      an attribute that also need custom unflattening, nest functions will
      utilize this method to do recursive unflattening.
    """