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
def _modify_model_input_type_per_subgraph(model, subgraph_index, signature_index, inference_input_type):
    """Modify model input type per subgraph."""
    subgraph = model.subgraphs[subgraph_index]
    tensors = subgraph.tensors
    operators = subgraph.operators
    quant_opcode_idxs = get_quantize_opcode_idx(model)
    if operators and (not quant_opcode_idxs):
        for input_idx in subgraph.inputs:
            input_type = _convert_tflite_enum_type_to_tf_type(tensors[input_idx].type)
            if input_type == dtypes.float32:
                raise ValueError('Model input is not dequantized.')
        return
    input_quant_ops = []
    for op in operators:
        if op.opcodeIndex in quant_opcode_idxs and op.inputs[0] in subgraph.inputs:
            float_tensor, quant_tensor = (tensors[op.inputs[0]], tensors[op.outputs[0]])
            float_type = _convert_tflite_enum_type_to_tf_type(float_tensor.type)
            if float_type != dtypes.float32:
                if float_type == inference_input_type:
                    continue
                else:
                    raise ValueError("Initial model input type must be tf.float32. Expected type for tensor with name '{}' is tf.float32, instead type is {}".format(float_tensor.name, get_tf_type_name(float_type)))
            quant_type = _convert_tflite_enum_type_to_tf_type(quant_tensor.type)
            if quant_type not in _MAP_QUANT_TO_IO_TYPES:
                raise ValueError("Initial model input is not quantized. Expected type for tensor with name '{}' should be in {}, instead type is {}".format(quant_tensor.name, tuple((get_tf_type_name(t) for t in _MAP_QUANT_TO_IO_TYPES.keys())), get_tf_type_name(quant_type)))
            else:
                inference_io_types = _MAP_QUANT_TO_IO_TYPES[quant_type]
                if inference_input_type not in inference_io_types:
                    raise ValueError('Unsupported `inference_input_type` value. Expected to be in {}, instead got {}.'.format(tuple((get_tf_type_name(t) for t in inference_io_types)), get_tf_type_name(inference_input_type)))
            input_quant_ops.append(op)
    if len(subgraph.inputs) != len(input_quant_ops):
        logging.warning('For model inputs containing unsupported operations which cannot be quantized, the `inference_input_type` attribute will default to the original type.')
    if inference_input_type == dtypes.uint8:
        for op in input_quant_ops:
            int8_quantization = tensors[op.outputs[0]].quantization
            uint8_quantization = schema_fb.QuantizationParametersT()
            uint8_quantization.scale = [int8_quantization.scale[0]]
            uint8_quantization.zeroPoint = [int8_quantization.zeroPoint[0] + 128]
            tensors[op.inputs[0]].quantization = uint8_quantization
            tensors[op.inputs[0]].type = schema_fb.TensorType.UINT8
    elif inference_input_type in _MAP_QUANT_TO_IO_TYPES:
        remove_tensors_idxs = set()
        for op in input_quant_ops:
            subgraph.inputs[subgraph.inputs == op.inputs[0]] = op.outputs[0]
            if signature_index >= 0:
                signature_def = model.signatureDefs[signature_index]
                for i in range(len(signature_def.inputs)):
                    if signature_def.inputs[i].tensorIndex == op.inputs[0]:
                        signature_def.inputs[i].tensorIndex = op.outputs[0]
            remove_tensors_idxs.add(op.inputs[0])
            operators.remove(op)
        _remove_tensors_from_model(model, remove_tensors_idxs)
    else:
        raise ValueError('Unsupported `inference_input_type` value {}.'.format(get_tf_type_name(inference_input_type)))