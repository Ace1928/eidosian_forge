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
def _run_impl(self, impl, *inputs, **kwargs):
    if len(impl.input_names) != len(inputs):
        raise RuntimeError(f'Mismatch lengths between the number of inputs {len(inputs)} and the expected number of inputs {len(impl.inputs)} for node {self.op_type!r} from domain {self.domain!r}.')
    feeds = dict(zip(impl.input_names, inputs))
    attributes = self.attributes_.copy()
    attributes.update(kwargs)
    results = impl.run(None, feeds, attributes=attributes)
    if len(impl.output_names) != len(results):
        raise RuntimeError(f'Mismatch lengths between the number of outputs {len(results)} and the expected number of outputs {len(impl.output_names)} for node {self.op_type!r} from domain {self.domain!r}.')
    return tuple(results)