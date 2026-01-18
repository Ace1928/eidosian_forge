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
class OpFunction(OpRun):
    """Runs a custom function."""

    def __init__(self, onnx_node: NodeProto, run_params: dict[str, Any] | None, impl: Any=None, attributes: dict[str, Any] | None=None):
        if impl is None:
            raise RuntimeError(f'impl cannot be None for node type {onnx_node.op_type!r} from domain {onnx_node.domain!r}.')
        OpRun.__init__(self, onnx_node, run_params)
        self.impl_ = impl
        self.attributes_ = {name: getattr(self, name) for name in getattr(self.impl_, 'attributes_', attributes)}

    def _run(self, *inputs, **kwargs):
        return self._run_impl(self.impl_, *inputs, **kwargs)

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