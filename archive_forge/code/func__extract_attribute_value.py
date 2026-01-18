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
def _extract_attribute_value(self, att: AttributeProto, ref_att: AttributeProto | None=None) -> Any:
    """Converts an attribute value into a python value."""
    if att.type == AttributeProto.GRAPH:
        new_ops = self.run_params.get('new_ops', None)
        if 'existing_functions' in self.run_params:
            functions = list(self.run_params['existing_functions'].values())
        else:
            functions = None
        evaluator_cls = self.run_params.get('evaluator_cls', None)
        assert evaluator_cls is not None, f'evaluator_cls must be specified to evaluate att={att}'
        return evaluator_cls(att.g, opsets=self.run_params['opsets'], verbose=max(0, self.run_params.get('verbose', 0) - 2), new_ops=None if new_ops is None else list(new_ops.values()), functions=functions)
    if att.type in OpRun._attribute_conversion_functions:
        return OpRun._attribute_conversion_functions[att.type](att)
    if ref_att is None:
        raise AttributeError(f'Unable to convert attribute {att.name!r} type {att.type!r} from node type {self.onnx_node.op_type!r}, domain {self.onnx_node.domain!r}\n{att}.')
    raise AttributeError(f'Unable to convert default value for {ref_att.name!r} type {att.type!r} from node type {self.onnx_node.op_type!r}, domain {self.onnx_node.domain!r}\n{att}\n{ref_att}.')