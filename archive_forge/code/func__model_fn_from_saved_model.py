from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import six
import tensorflow as tf
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_constants
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _model_fn_from_saved_model(self, features, labels, mode):
    """Load a SavedModel graph and return an EstimatorSpec."""
    self._validate_mode(mode)
    g = tf.compat.v1.get_default_graph()
    if tf.compat.v1.train.get_global_step(g) is not None:
        raise RuntimeError('Graph must not contain a global step tensor before the SavedModel is loaded. Please make sure that the input function does not create a global step.')
    signature_def = self._get_signature_def_for_mode(mode)
    input_map = _generate_input_map(signature_def, features, labels)
    output_tensor_names = [value.name for value in six.itervalues(signature_def.outputs)]
    tags = export_lib.EXPORT_TAG_MAP[mode]
    _, output_tensors = self.saved_model_loader.load_graph(g, tags, input_map=input_map, return_elements=output_tensor_names)
    saver_obj = tf.compat.v1.train.Saver(saver_def=self._get_saver_def_from_mode(mode))
    init_fn = None
    if not super(SavedModelEstimator, self).latest_checkpoint():
        init_fn = self._restore_from_saver
    meta_graph_def = self._get_meta_graph_def_for_mode(mode)
    asset_tensors_dictionary = loader_impl.get_asset_tensors(self.saved_model_loader.export_dir, meta_graph_def, import_scope=None)
    scaffold = tf.compat.v1.train.Scaffold(local_init_op=loader_impl._get_main_op_tensor(meta_graph_def), local_init_feed_dict=asset_tensors_dictionary, saver=saver_obj, init_fn=init_fn)
    global_step_tensor = tf.compat.v1.train.get_global_step(g)
    tf.compat.v1.train.assert_global_step(global_step_tensor)
    output_map = dict(zip(output_tensor_names, output_tensors))
    outputs = {key: output_map[value.name] for key, value in six.iteritems(signature_def.outputs)}
    loss, predictions, metrics = _validate_and_extract_outputs(mode, outputs, signature_def.method_name)
    train_op = tf.compat.v1.get_collection(constants.TRAIN_OP_KEY)
    if len(train_op) > 1:
        raise RuntimeError('Multiple ops found in the train_op collection.')
    train_op = None if not train_op else train_op[0]
    _clear_saved_model_collections()
    return model_fn_lib.EstimatorSpec(scaffold=scaffold, mode=mode, loss=loss, train_op=train_op, predictions=predictions, eval_metric_ops=metrics)