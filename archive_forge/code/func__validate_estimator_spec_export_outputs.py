from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _validate_estimator_spec_export_outputs(export_outputs, predictions, mode):
    """Validate export_outputs inputs for EstimatorSpec or TPUEstimatorSpec.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving.
      A dict `{name: output}` where:
      * name: An arbitrary name for this output.
      * output: an `ExportOutput` object such as `ClassificationOutput`
        `RegressionOutput`, or `PredictOutput`. Single-headed models should only
        need to specify one entry in this dictionary. Multi-headed models should
        specify one entry for each head, one of which must be named using
        `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`.
        If no entry is provided, a default `PredictOutput` mapping to
        predictions will be created.
    predictions: Predictions `Tensor` or dict of `Tensor`. Used in generation of
      default outputs.
    mode: A `ModeKeys`. Used to determine whether to validate at all; if the
      EstimatorSpec is not for making predictions we can skip validation.

  Returns:
    ValueError: If validation fails.
    TypeError: If the export_outputs is not a dict or the values of the dict are
               not instances of type `ExportOutput`.
  """
    if mode == ModeKeys.PREDICT:
        export_outputs = export_utils.get_export_outputs(export_outputs, predictions)
    return export_outputs