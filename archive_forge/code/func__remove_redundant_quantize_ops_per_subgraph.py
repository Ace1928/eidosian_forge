import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def _remove_redundant_quantize_ops_per_subgraph(model, subgraph_index, signature_index):
    """Remove redundant quantize ops per subgraph."""
    subgraph = model.subgraphs[subgraph_index]
    tensors = subgraph.tensors
    operators = subgraph.operators
    quant_opcode_idxs = get_quantize_opcode_idx(model)
    dequant_opcode_idxs = get_dequantize_opcode_idx(model)
    all_quant_ops = []
    redundant_quant_tensors = {}
    output_dequant_tensors = {}
    for op in operators:
        if op.opcodeIndex in quant_opcode_idxs:
            all_quant_ops.append(op)
            input_tensor = tensors[op.inputs[0]]
            output_tensor = tensors[op.outputs[0]]
            input_type = _convert_tflite_enum_type_to_tf_type(input_tensor.type)
            output_type = _convert_tflite_enum_type_to_tf_type(output_tensor.type)
            if input_type != dtypes.float32 and output_type != dtypes.float32:
                redundant_quant_tensors[op.inputs[0]] = op
        if op.opcodeIndex in dequant_opcode_idxs and op.outputs[0] in subgraph.outputs:
            output_dequant_tensors[op.inputs[0]] = op
    for op in all_quant_ops:
        output_tensor_idx = op.outputs[0]
        if output_tensor_idx in redundant_quant_tensors:
            requantize_op = redundant_quant_tensors[output_tensor_idx]
            if model.signatureDefs:
                signature_def = model.signatureDefs[0]
                for output in signature_def.outputs:
                    if output.tensorIndex == op.outputs[0]:
                        output.tensorIndex = op.inputs[0]
            requantize_op.inputs[0] = op.inputs[0]
            operators.remove(op)
    for op in all_quant_ops:
        output_tensor_idx = op.outputs[0]
        if output_tensor_idx in output_dequant_tensors:
            dequant_op = output_dequant_tensors[output_tensor_idx]
            subgraph.outputs[subgraph.outputs == dequant_op.outputs[0]] = op.inputs[0]
            if signature_index >= 0:
                signature_def = model.signatureDefs[signature_index]
                for output in signature_def.outputs:
                    if output.tensorIndex == dequant_op.outputs[0]:
                        output.tensorIndex = op.inputs[0]
            operators.remove(op)
            operators.remove(dequant_op)