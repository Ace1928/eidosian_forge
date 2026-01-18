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
class Sentinel(object):
    """Sentinel value that's not equal (w/ `is`) to any user value."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name