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
def __tf_flatten__(self):
    """Flatten current object into (metadata, components).

    Returns:
      A `tuple` of (metadata, components), where
        - metadata is a custom Python object that stands for the static config
          of the current object, which is supposed to be fixed and not affected
          by data transformation.
        - components is a `tuple` that contains the modifiable fields of the
          current object.

    Implementation Note:
    - This method should not invoke any TensorFlow ops.
    - This method only needs to flatten the current level. If current object has
      an attribute that also need custom flattening, nest functions (such as
      `nest.flatten`) will utilize this method to do recursive flattening.
    - Components must ba a `tuple`, not a `list`
    """