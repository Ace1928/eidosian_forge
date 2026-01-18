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
def _export_mode(mode, has_saved_vars, builder, model, custom_objects, checkpoint_path, input_signature):
    """Exports a model, and optionally saves new vars from the clone model.

  Args:
    mode: A `tf.estimator.ModeKeys` string.
    has_saved_vars: A `boolean` indicating whether the SavedModel has already
      exported variables.
    builder: A `SavedModelBuilder` object.
    model: A `tf.keras.Model` object.
    custom_objects: A dictionary mapping string names to custom classes
      or functions.
    checkpoint_path: String path to checkpoint.
    input_signature: Nested TensorSpec containing the expected inputs. Can be
      `None`, in which case the signature will be inferred from the model.

  Raises:
    ValueError: If the train/eval mode is being exported, but the model does
      not have an optimizer.
  """
    compile_clone = mode != mode_keys.ModeKeys.PREDICT
    if compile_clone and (not model.optimizer):
        raise ValueError('Model does not have an optimizer. Cannot export mode %s' % mode)
    model_graph = ops.get_default_graph()
    with ops.Graph().as_default() as g, backend.learning_phase_scope(mode == mode_keys.ModeKeys.TRAIN):
        if input_signature is None:
            input_tensors = None
        else:
            input_tensors = nest.map_structure(create_placeholder, input_signature)
        clone = models_lib.clone_and_build_model(model, input_tensors=input_tensors, custom_objects=custom_objects, compile_clone=compile_clone)
        if compile_clone:
            g.add_to_collection(ops.GraphKeys.GLOBAL_STEP, clone.optimizer.iterations)
        train_op = None
        if mode == mode_keys.ModeKeys.TRAIN:
            clone._make_train_function()
            train_op = clone.train_function.updates_op
        elif mode == mode_keys.ModeKeys.TEST:
            clone._make_test_function()
        else:
            clone._make_predict_function()
        g.get_collection_ref(ops.GraphKeys.UPDATE_OPS).extend(clone.state_updates)
        with session.Session().as_default():
            clone_var_list = _get_var_list(clone)
            if has_saved_vars:
                status = clone.load_weights(checkpoint_path)
                status.assert_existing_objects_matched()
            else:
                _assert_same_non_optimizer_objects(model, model_graph, clone, g)
                clone.load_weights(checkpoint_path)
                clone.save_weights(checkpoint_path, save_format='tf', overwrite=True)
                builder._has_saved_variables = True
            builder.add_meta_graph(model_utils.EXPORT_TAG_MAP[mode], signature_def_map=_create_signature_def_map(clone, mode), saver=saver_lib.Saver(clone_var_list, allow_empty=True), init_op=variables.local_variables_initializer(), train_op=train_op)
        return None