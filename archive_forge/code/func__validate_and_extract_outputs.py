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
def _validate_and_extract_outputs(mode, output_dict, method_name):
    """Extract values from SignatureDef output dictionary.

  Args:
    mode: One of the modes enumerated in `tf.estimator.ModeKeys`.
    output_dict: dict of string SignatureDef keys to `Tensor`.
    method_name: Method name of the SignatureDef as a string.

  Returns:
    Tuple of (
      loss: `Tensor` object,
      predictions: dictionary mapping string keys to `Tensor` objects,
      metrics: dictionary mapping string keys to a tuple of two `Tensor` objects
    )

  Raises:
    RuntimeError: raised if SignatureDef has an invalid method name for the mode
  """
    loss, predictions, metrics = (None, None, None)
    if mode == ModeKeys.PREDICT:
        predictions = output_dict
    else:
        expected_method_name = signature_constants.SUPERVISED_TRAIN_METHOD_NAME
        if mode == ModeKeys.EVAL:
            expected_method_name = signature_constants.SUPERVISED_EVAL_METHOD_NAME
        if method_name != expected_method_name:
            raise RuntimeError('Invalid SignatureDef method name for mode %s.\n\tExpected: %s\n\tGot: %s\nPlease ensure that the SavedModel was exported with `tf.estimator.experimental_export_all_saved_models()`.' % (mode, expected_method_name, method_name))
        loss = output_dict[export_lib._SupervisedOutput.LOSS_NAME]
        metrics = _extract_eval_metrics(output_dict)
        predictions = {key: value for key, value in six.iteritems(output_dict) if key.split(export_lib._SupervisedOutput._SEPARATOR_CHAR)[0] == export_lib._SupervisedOutput.PREDICTIONS_NAME}
    return (loss, predictions, metrics)