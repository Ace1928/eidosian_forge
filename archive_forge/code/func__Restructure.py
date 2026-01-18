from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _Restructure(l, structure):
    """Returns the elements of list l structured according to the given structure.

  A structure is represented by a list whose elements are either
  `None` or a non-negative integer. `None` corresponds to a single
  element in the output list, and an integer N corresponds to a nested
  list of length N.

  The function returns a data structure whose shape is given by
  `structure`, and whose elements are taken from `l`. If `structure`
  is a singleton, the function returns the single data structure
  implied by the 0th element of `structure`. For example:

      _Restructure(["foo", "bar", "baz", "qux"], [None, 2, None])
        -> ["foo", ["bar", "baz"], "qux"]

      _Restructure(["foo"], [None]) -> "foo"

      _Restructure(["foo"], [1]) -> ["foo"]

      _Restructure([], [0]) -> []

  Args:
    l: A list.
    structure: A list whose elements are either `None` or a non-negative
      integer.

  Returns:
    The elements of `l`, restructured according to `structure`. If
    `structure` is a list of length 1, this function returns the
    single data structure implied by `structure[0]`.

  """
    result = []
    current_index = 0
    for element in structure:
        if element is None:
            result.append(l[current_index])
            current_index += 1
        else:
            result.append(l[current_index:current_index + element])
            current_index += element
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)