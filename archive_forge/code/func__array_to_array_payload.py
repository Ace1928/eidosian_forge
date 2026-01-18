from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    """Serialize an Arrow Array to an PicklableArrayPayload for later pickling.

    This function's primary purpose is to dispatch to the handler for the input array
    type.
    """
    import pyarrow as pa
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType, ArrowVariableShapedTensorType
    if _is_dense_union(a.type):
        raise NotImplementedError('Custom slice view serialization of dense union arrays is not yet supported.')
    if pa.types.is_null(a.type):
        return _null_array_to_array_payload(a)
    elif _is_primitive(a.type):
        return _primitive_array_to_array_payload(a)
    elif _is_binary(a.type):
        return _binary_array_to_array_payload(a)
    elif pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
        return _list_array_to_array_payload(a)
    elif pa.types.is_fixed_size_list(a.type):
        return _fixed_size_list_array_to_array_payload(a)
    elif pa.types.is_struct(a.type):
        return _struct_array_to_array_payload(a)
    elif pa.types.is_union(a.type):
        return _union_array_to_array_payload(a)
    elif pa.types.is_dictionary(a.type):
        return _dictionary_array_to_array_payload(a)
    elif pa.types.is_map(a.type):
        return _map_array_to_array_payload(a)
    elif isinstance(a.type, ArrowTensorType) or isinstance(a.type, ArrowVariableShapedTensorType):
        return _tensor_array_to_array_payload(a)
    else:
        raise ValueError('Unhandled Arrow array type:', a.type)