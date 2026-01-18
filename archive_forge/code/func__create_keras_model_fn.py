from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _create_keras_model_fn(keras_model, custom_objects=None, save_object_ckpt=False, metric_names_map=None, export_outputs=None):
    """Creates model_fn for keras Estimator.

  Args:
    keras_model: an instance of compiled keras model.
    custom_objects: Dictionary for custom objects.
    save_object_ckpt: Whether to save an object-based checkpoint.
    metric_names_map: Optional dictionary mapping Keras model output metric
      names to custom names.
    export_outputs: Optional dictionary mapping custom names to a subclass of
      `tf.estimator.export.ExportOutput`.

  Returns:
    The model_fn for a keras Estimator.
  """
    if isinstance(keras_model.optimizer, tf.keras.optimizers.experimental.Optimizer):
        if tf.executing_eagerly():
            logging.warning('You are using `tf.keras.optimizers.experimental.Optimizer` in TF estimator, which only supports `tf.keras.optimizers.legacy.Optimizer`. Automatically converting your optimizer to `tf.keras.optimizers.legacy.Optimizer`.')
            opt = tf.keras.__internal__.optimizers.convert_to_legacy_optimizer(keras_model.optimizer)
            keras_model.optimizer = opt
        else:
            raise ValueError(f'Please set your optimizer as an instance of `tf.keras.optimizers.legacy.Optimizer`, e.g., `tf.keras.optimizers.legacy.Adam`. Received optimizer type: {type(keras_model.optimizer)}.')
    try:
        if isinstance(keras_model.optimizer, (tuple, list)):
            optimizer_config = [opt.get_config() for opt in keras_model.optimizer]
        else:
            optimizer_config = keras_model.optimizer.get_config()
    except (NotImplementedError, AttributeError):
        optimizer_config = None

    def model_fn(features, labels, mode):
        """model_fn for keras Estimator."""
        model = _clone_and_build_model(mode=mode, keras_model=keras_model, custom_objects=custom_objects, features=features, labels=labels, optimizer_config=optimizer_config)
        model_output_names = []
        if tf.distribute.has_strategy():
            for name in model.output_names:
                name = re.compile('_\\d$').sub('', name)
                model_output_names.append(name)
        else:
            model_output_names = model.output_names
        predictions = dict(zip(model_output_names, model.outputs))
        loss = None
        train_op = None
        eval_metric_ops = None
        if mode is not ModeKeys.PREDICT:
            if mode is ModeKeys.TRAIN:
                model._make_train_function()
            else:
                model._make_test_function()
            loss = model.total_loss
            eval_metric_ops = _convert_keras_metrics_to_estimator(model, metric_names_map)
        if mode is ModeKeys.TRAIN:
            train_op = model.train_function.updates_op
        if not model._is_graph_network and hasattr(keras_model, '_original_attributes_cache') and (keras_model._original_attributes_cache is not None):
            tf.compat.v2.keras.__internal__.models.in_place_subclassed_model_state_restoration(keras_model)
        scaffold = None
        if save_object_ckpt:
            model._track_trackable(tf.compat.v1.train.get_global_step(), 'estimator_global_step')
            object_graph = tf.compat.v2.__internal__.tracking.ObjectGraphView(model)
            var_list = object_graph.frozen_saveable_objects()
            saver = tf.compat.v1.train.Saver(var_list=var_list, sharded=True)
            saver._object_restore_saver = trackable_util.frozen_saver(model)
            scaffold = tf.compat.v1.train.Scaffold(saver=saver)
        final_export_outputs = {_DEFAULT_SERVING_KEY: export_lib.PredictOutput(predictions)}
        if export_outputs is not None:
            different_keys = set(export_outputs.keys()) - set(model.output_names)
            if different_keys:
                raise FormattedKeyError('The list passed into {obj_name} does not cover requested {order_name} keys defined in the keras model.\n\tExpected keys: {order_keys}\n\t{obj_name} keys: {obj_keys}\n\tMissed keys: {different_keys}'.format(order_name=export_outputs, order_keys=set(export_outputs.keys()), obj_name=model.output_names, obj_keys=set(model.output_names), different_keys=different_keys))
            for key, export_output_cls in export_outputs.items():
                final_export_outputs[key] = export_output_cls(predictions[key])
        return model_fn_lib.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, export_outputs=final_export_outputs, scaffold=scaffold)
    return model_fn