import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
def _load_tf_custom_op(model_path):
    """Loads a custom TF OP (in .so format) from /assets.extra directory."""
    assets_dir = os.path.join(model_path, _CUSTOM_OP_DIRECTORY_NAME)
    if tf.gfile.IsDirectory(assets_dir):
        custom_ops_pattern = os.path.join(assets_dir, _CUSTOM_OP_SUFFIX)
        for custom_op_path_original in tf.gfile.Glob(custom_ops_pattern):
            logging.info('Found custom op file: %s', custom_op_path_original)
            if custom_op_path_original.startswith('gs://'):
                if not os.path.isdir(_CUSTOM_OP_LOCAL_DIR):
                    os.makedirs(_CUSTOM_OP_LOCAL_DIR)
                custom_op_path_local = os.path.join(_CUSTOM_OP_LOCAL_DIR, os.path.basename(custom_op_path_original))
                logging.info('Copying custop op from: %s to: %s', custom_op_path_original, custom_op_path_local)
                tf.gfile.Copy(custom_op_path_original, custom_op_path_local, True)
            else:
                custom_op_path_local = custom_op_path_original
            try:
                logging.info('Loading custom op: %s', custom_op_path_local)
                logging.info('TF Version: %s', tf.__version__)
                tf.load_op_library(custom_op_path_local)
            except RuntimeError as e:
                logging.exception('Failed to load custom op: %s with error: %s. Prediction will likely fail due to missing operations.', custom_op_path_local, e)