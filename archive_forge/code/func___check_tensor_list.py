import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings
import numpy as np
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def __check_tensor_list(self, tensor_list):
    """Raises an exception if tensor_list incompatible w/ flat_tensor_specs."""
    expected = self._flat_tensor_specs
    specs = [type_spec_from_value(t) for t in tensor_list]
    if len(specs) != len(expected):
        raise ValueError(f'Cannot create a {self.value_type.__name__} from the tensor list because the TypeSpec expects {len(expected)} items, but the provided tensor list has {len(specs)} items.')
    for i, (s1, s2) in enumerate(zip(specs, expected)):
        if not s1.is_compatible_with(s2):
            raise ValueError(f'Cannot create a {self.value_type.__name__} from the tensor list because item {i} ({tensor_list[i]!r}) is incompatible with the expected TypeSpec {s2}.')