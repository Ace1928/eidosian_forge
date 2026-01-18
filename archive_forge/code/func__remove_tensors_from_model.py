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
def _remove_tensors_from_model(model, remove_tensors_idxs):
    """Remove tensors from model."""
    if not remove_tensors_idxs:
        return
    if len(model.subgraphs) > 1:
        logging.info('Skipping the removal of dangled tensors since the model has multiple subgraphs and tensors can be used in the different subgraph(s)')
        return
    subgraph = model.subgraphs[0]
    tensors = subgraph.tensors
    operators = subgraph.operators
    logging.debug('Removing tensors at indices : %s', remove_tensors_idxs)
    if min(remove_tensors_idxs) == len(tensors) - len(remove_tensors_idxs):
        logging.debug('Removing tensors only at the end of the tensor list')
        del tensors[min(remove_tensors_idxs):]
    else:
        logging.debug('Removing tensors requires updating the model')
        d_old_to_new_tensors = {}
        left_shift_by = 0
        for idx in range(len(tensors)):
            if idx in remove_tensors_idxs:
                left_shift_by += 1
            else:
                d_old_to_new_tensors[idx] = idx - left_shift_by
        logging.debug('Old to new tensors map: %s', d_old_to_new_tensors.__str__())

        def update_tensors(tensor_idxs):
            for i, ti in enumerate(tensor_idxs):
                tensor_idxs[i] = d_old_to_new_tensors.get(ti, -1)
        update_tensors(subgraph.inputs)
        update_tensors(subgraph.outputs)
        for op in operators:
            update_tensors(op.inputs)
            update_tensors(op.outputs)
        if model.signatureDefs:
            signature_def = model.signatureDefs[0]
            _update_signature_def_tensors(signature_def.inputs, d_old_to_new_tensors)
            _update_signature_def_tensors(signature_def.outputs, d_old_to_new_tensors)
        for idx in sorted(remove_tensors_idxs, reverse=True):
            tensors.pop(idx)
        logging.debug('Removed tensors marked for deletion')