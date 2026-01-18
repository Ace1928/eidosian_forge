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
def _get_file_from_google_storage(keras_model_path, model_dir):
    """Get file from google storage and download to local file.

  Args:
    keras_model_path: a google storage path for compiled keras model.
    model_dir: the directory from estimator config.

  Returns:
    The path where keras model is saved.

  Raises:
    ValueError: if storage object name does not end with .h5.
  """
    try:
        from google.cloud import storage
    except ImportError:
        raise TypeError('Could not save model to Google cloud storage; please install `google-cloud-storage` via `pip install google-cloud-storage`.')
    storage_client = storage.Client()
    path, blob_name = os.path.split(keras_model_path)
    _, bucket_name = os.path.split(path)
    keras_model_dir = os.path.join(model_dir, 'keras')
    if not tf.compat.v1.gfile.Exists(keras_model_dir):
        tf.compat.v1.gfile.MakeDirs(keras_model_dir)
    file_name = os.path.join(keras_model_dir, 'keras_model.h5')
    try:
        blob = storage_client.get_bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(file_name)
    except:
        raise ValueError('Failed to download keras model, please check environment variable GOOGLE_APPLICATION_CREDENTIALS and model path storage.googleapis.com/{bucket}/{object}.')
    tf.compat.v1.logging.info('Saving model to {}'.format(file_name))
    del storage_client
    return file_name