from __future__ import annotations
import abc
from typing import Any, ClassVar, Iterable
import numpy as np
from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history, get_schema, onnx_opset_version
from onnx.helper import make_node, make_tensor_type_proto, np_dtype_to_tensor_dtype
from onnx.numpy_helper import to_array, unpack_int4
from onnx.onnx_pb import AttributeProto, GraphProto, NodeProto, TypeProto
from onnx.reference.custom_element_types import (
def _check_and_fix_outputs(self, res: tuple[Any, ...]) -> tuple[Any, ...]:
    """Checks the output are from the expected type."""
    if not isinstance(res, tuple):
        raise TypeError(f"Method '_run' of class {self.__class__.__name__!r} does not return a tuple but '{type(res)}'.")
    if not res:
        raise ValueError(f"Method '_run' of class {self.__class__.__name__!r} does not return any result.")
    if any((isinstance(t, tuple) for t in res)):
        dtypes = [type(t) for t in res]
        raise TypeError(f"One of the results returned by method '_run' of class {self.__class__.__name__!r} is a tuple, this is no ONNX corresponding type (Map, List, Tensor, SparseTensor). All returned types: {dtypes!r}.")
    res = tuple((np.array(x) if np.isscalar(x) else x for x in res))
    if any((not (isinstance(t, (np.ndarray, list, dict)) or hasattr(t, 'todense')) for t in res)):
        dtypes = [type(t) for t in res]
        raise TypeError(f"One of the results returned by method '_run' of class {self.__class__.__name__!r} has an unexpected type, this is no ONNX correponding type (Map, List, Tensor, SparseTensor). All returned types: {dtypes!r}.")
    return res