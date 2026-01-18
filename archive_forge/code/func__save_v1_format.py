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
def _save_v1_format(model, path, custom_objects, as_text, input_signature):
    """Exports model to v1 SavedModel format."""
    if not model._is_graph_network:
        if isinstance(model, sequential.Sequential):
            if not model.built:
                raise ValueError('Weights for sequential model have not yet been created. Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`, or the first layer in the model has `input_shape` during construction.')
        else:
            raise NotImplementedError('Subclassed models can only be exported for serving. Please set argument serving_only=True.')
    builder = saved_model_builder._SavedModelBuilder(path)
    checkpoint_path = _export_model_variables(model, path)
    export_args = {'builder': builder, 'model': model, 'custom_objects': custom_objects, 'checkpoint_path': checkpoint_path, 'input_signature': input_signature}
    has_saved_vars = False
    if model.optimizer:
        if isinstance(model.optimizer, (optimizer_v1.TFOptimizer, optimizer_v2.OptimizerV2)):
            _export_mode(mode_keys.ModeKeys.TRAIN, has_saved_vars, **export_args)
            has_saved_vars = True
            _export_mode(mode_keys.ModeKeys.TEST, has_saved_vars, **export_args)
        else:
            logging.warning('Model was compiled with an optimizer, but the optimizer is not from `tf.train` (e.g. `tf.train.AdagradOptimizer`). Only the serving graph was exported. The train and evaluate graphs were not added to the SavedModel.')
    _export_mode(mode_keys.ModeKeys.PREDICT, has_saved_vars, **export_args)
    builder.save(as_text)