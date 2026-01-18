import collections
import collections.abc
import enum
import typing
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations
class _ConversionContext(enum.Enum):
    """Enum to indicate what kind of value is being converted.

  Used by `_convert_fields` and `_convert_value` and their helper methods.
  """
    VALUE = 1
    SPEC = 2
    DEFAULT = 3