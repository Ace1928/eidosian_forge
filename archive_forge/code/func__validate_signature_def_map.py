import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _validate_signature_def_map(self, signature_def_map):
    """Validates the `SignatureDef` entries in the signature def map.

    Validation of entries in the signature def map includes ensuring that the
    `name` and `dtype` fields of the TensorInfo protos of the `inputs` and
    `outputs` of each `SignatureDef` are populated. Also ensures that reserved
    SignatureDef keys for the initialization and train ops are not used.

    Args:
      signature_def_map: The map of signature defs to be validated.

    Raises:
      AssertionError: If a TensorInfo is not valid.
      KeyError: If a reserved signature key is used in the map.
    """
    for signature_def_key in signature_def_map:
        signature_def = signature_def_map[signature_def_key]
        inputs = signature_def.inputs
        outputs = signature_def.outputs
        for inputs_key in inputs:
            self._validate_tensor_info(inputs[inputs_key])
        for outputs_key in outputs:
            self._validate_tensor_info(outputs[outputs_key])
    if constants.INIT_OP_SIGNATURE_KEY in signature_def_map:
        raise KeyError(f'SignatureDef map key "{constants.INIT_OP_SIGNATURE_KEY}" is reserved for initialization. Please use a different key.')
    if constants.TRAIN_OP_SIGNATURE_KEY in signature_def_map:
        raise KeyError(f'SignatureDef map key "{constants.TRAIN_OP_SIGNATURE_KEY}" is reserved for the train op. Please use a different key.')