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
class OpFunctionContextDependant(OpFunction):
    """The function can be instantiated but only at execution time.
    An instance of OpFunction is created everytime to node is executed.
    This is needed when the schema of an operator defines a context dependant function.
    """

    def __init__(self, onnx_node: NodeProto, run_params: dict[str, Any] | None, parent: Any=None):
        OpFunction.__init__(self, onnx_node, run_params, impl=self, attributes={})
        self.parent = parent
        version = parent.opsets[onnx_node.domain]
        self.schema_ = get_schema(onnx_node.op_type, version, onnx_node.domain)

    def _run(self, *inputs, **kwargs):
        types = []
        for t in inputs:
            try:
                ttype = np_dtype_to_tensor_dtype(t.dtype)
            except KeyError as e:
                if t.dtype == float8e4m3fn:
                    ttype = TensorProto.FLOAT8E4M3FN
                elif t.dtype == float8e4m3fnuz:
                    ttype = TensorProto.FLOAT8E4M3FNUZ
                elif t.dtype == float8e5m2:
                    ttype = TensorProto.FLOAT8E5M2
                elif t.dtype == float8e5m2fnuz:
                    ttype = TensorProto.FLOAT8E5M2FNUZ
                elif t.dtype == bfloat16:
                    ttype = TensorProto.BLOFAT16
                elif t.dtype == uint4:
                    ttype = TensorProto.UINT4
                elif t.dtype == int4:
                    ttype = TensorProto.INT4
                else:
                    raise e
            types.append(make_tensor_type_proto(ttype, t.shape))
        cl = self.parent._load_impl(self.onnx_node, types)
        inst = cl(self.onnx_node, self.run_params)
        return self._run_impl(inst.impl_, *inputs, **kwargs)