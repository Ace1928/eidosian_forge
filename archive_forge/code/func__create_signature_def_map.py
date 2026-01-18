import os
import warnings
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving import utils_v1 as model_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def _create_signature_def_map(model, mode):
    """Creates a SignatureDef map from a Keras model."""
    inputs_dict = {name: x for name, x in zip(model.input_names, model.inputs)}
    if model.optimizer:
        targets_dict = {x.name.split(':')[0]: x for x in model._targets if x is not None}
        inputs_dict.update(targets_dict)
    outputs_dict = {name: x for name, x in zip(model.output_names, model.outputs)}
    metrics = saving_utils.extract_model_metrics(model)
    local_vars = set(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
    vars_to_add = set()
    if metrics is not None:
        for key, value in metrics.items():
            if isinstance(value, metrics_lib.Metric):
                vars_to_add.update(value.variables)
                metrics[key] = (value.result(), value.updates[0])
    vars_to_add = vars_to_add.difference(local_vars)
    for v in vars_to_add:
        ops.add_to_collection(ops.GraphKeys.LOCAL_VARIABLES, v)
    export_outputs = model_utils.export_outputs_for_mode(mode, predictions=outputs_dict, loss=model.total_loss if model.optimizer else None, metrics=metrics)
    return model_utils.build_all_signature_defs(inputs_dict, export_outputs=export_outputs, serving_only=mode == mode_keys.ModeKeys.PREDICT)