import numbers as _numbers
import numpy as _np
import six as _six
import codecs
from tensorflow.python.util.tf_export import tf_export
Converts input which is a `PathLike` object to `bytes`.

  Converts from any python constant representation of a `PathLike` object
  or `str` to bytes.

  Args:
    path: An object that can be converted to path representation.

  Returns:
    A `bytes` object.

  Usage:
    In case a simplified `bytes` version of the path is needed from an
    `os.PathLike` object.
  