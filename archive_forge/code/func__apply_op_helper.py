from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _apply_op_helper(op_type_name, name=None, **keywords):
    """Implementation of apply_op that returns output_structure, op."""
    op_def, g, producer = _GetOpDef(op_type_name, keywords)
    name = name if name else op_type_name
    attrs, attr_protos = ({}, {})
    default_type_attr_map, allowed_list_attr_map = ({}, {})
    inputs, input_types, output_structure = ([], [], [])
    fallback = True
    if _CanExtractAttrsFastPath(op_def, keywords) and flags.config().graph_building_optimization.value():
        fallback = False
        attr_protos, inputs, input_types, output_structure = op_def_library_pybind.process_inputs(op_type_name, producer, keywords)
    if fallback:
        _CheckOpDeprecation(op_type_name, op_def, producer)
        _ExtractDefaultTypesAndAllowedTypes(op_def, default_type_attr_map, allowed_list_attr_map)
    with g.as_default(), ops.name_scope(name) as scope:
        if fallback:
            _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
            _ExtractRemainingAttrs(op_type_name, op_def, keywords, default_type_attr_map, attrs)
            _ExtractAttrProto(op_type_name, op_def, attrs, attr_protos)
            del attrs
            _ExtractOutputStructure(op_type_name, op_def, attr_protos, output_structure)
            _CheckAllInputsUsed(op_type_name, keywords)
        must_colocate_inputs = [val for arg, val in zip(op_def.input_arg, inputs) if arg.is_ref]
        with _MaybeColocateWith(must_colocate_inputs):
            op = g._create_op_internal(op_type_name, inputs, dtypes=None, name=scope, input_types=input_types, attrs=attr_protos, op_def=op_def)
        outputs = op.outputs
        if op_callbacks.should_invoke_op_callbacks():
            callback_outputs = op_callbacks.invoke_op_callbacks(op.node_def.op, tuple(op.inputs), attr_protos, tuple(outputs), op_name=op.name, graph=g)
            if callback_outputs is not None:
                outputs = callback_outputs
        return (output_structure, op_def.is_stateful, op, outputs)