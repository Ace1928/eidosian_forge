import collections
import csv
import functools
import gzip
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export
def _infer_type(str_val, na_value, prev_type):
    """Given a string, infers its tensor type.

  Infers the type of a value by picking the least 'permissive' type possible,
  while still allowing the previous type inference for this column to be valid.

  Args:
    str_val: String value to infer the type of.
    na_value: Additional string to recognize as a NA/NaN CSV value.
    prev_type: Type previously inferred based on values of this column that
      we've seen up till now.
  Returns:
    Inferred dtype.
  """
    if str_val in ('', na_value):
        return prev_type
    type_list = [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.string]
    type_functions = [_is_valid_int32, _is_valid_int64, lambda str_val: _is_valid_float(str_val, dtypes.float32), lambda str_val: _is_valid_float(str_val, dtypes.float64), lambda str_val: True]
    for i in range(len(type_list)):
        validation_fn = type_functions[i]
        if validation_fn(str_val) and (prev_type is None or prev_type in type_list[:i + 1]):
            return type_list[i]