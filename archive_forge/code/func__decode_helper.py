import collections
import enum
import json
import numpy as np
import wrapt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
def _decode_helper(obj):
    """A decoding helper that is TF-object aware."""
    if isinstance(obj, dict) and 'class_name' in obj:
        if obj['class_name'] == 'TensorShape':
            return tensor_shape.TensorShape(obj['items'])
        elif obj['class_name'] == 'TypeSpec':
            return type_spec_registry.lookup(obj['type_spec'])._deserialize(_decode_helper(obj['serialized']))
        elif obj['class_name'] == '__tuple__':
            return tuple((_decode_helper(i) for i in obj['items']))
        elif obj['class_name'] == '__ellipsis__':
            return Ellipsis
    return obj