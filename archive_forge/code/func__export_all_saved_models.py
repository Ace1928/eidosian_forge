from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _export_all_saved_models(self, export_dir_base, input_receiver_fn_map, assets_extra=None, as_text=False, checkpoint_path=None, strip_default_attrs=True):
    """Exports multiple modes in the model function to a SavedModel."""
    with context.graph_mode():
        if not checkpoint_path:
            checkpoint_path = self.latest_checkpoint()
        if not checkpoint_path:
            if self._warm_start_settings:
                checkpoint_path = self._warm_start_settings.ckpt_to_initialize_from
                if tf.compat.v1.gfile.IsDirectory(checkpoint_path):
                    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            else:
                raise ValueError("Couldn't find trained model at {}.".format(self._model_dir))
        export_dir = export_lib.get_timestamped_export_dir(export_dir_base)
        temp_export_dir = export_lib.get_temp_export_dir(export_dir)
        builder = tf.compat.v1.saved_model.Builder(temp_export_dir)
        save_variables = True
        if input_receiver_fn_map.get(ModeKeys.TRAIN):
            self._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables, mode=ModeKeys.TRAIN, strip_default_attrs=strip_default_attrs)
            save_variables = False
        if input_receiver_fn_map.get(ModeKeys.EVAL):
            self._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables, mode=ModeKeys.EVAL, strip_default_attrs=strip_default_attrs)
            save_variables = False
        if input_receiver_fn_map.get(ModeKeys.PREDICT):
            self._add_meta_graph_for_mode(builder, input_receiver_fn_map, checkpoint_path, save_variables, mode=ModeKeys.PREDICT, strip_default_attrs=strip_default_attrs)
            save_variables = False
        if save_variables:
            raise ValueError('No valid modes for exporting found. Got {}.'.format(input_receiver_fn_map.keys()))
        builder.save(as_text)
        if assets_extra:
            assets_extra_path = os.path.join(tf.compat.as_bytes(temp_export_dir), tf.compat.as_bytes('assets.extra'))
            for dest_relative, source in assets_extra.items():
                dest_absolute = os.path.join(tf.compat.as_bytes(assets_extra_path), tf.compat.as_bytes(dest_relative))
                dest_path = os.path.dirname(dest_absolute)
                tf.compat.v1.gfile.MakeDirs(dest_path)
                tf.compat.v1.gfile.Copy(source, dest_absolute)
        tf.compat.v1.gfile.Rename(temp_export_dir, export_dir)
        return export_dir