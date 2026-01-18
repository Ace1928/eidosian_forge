from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export('estimator.export.TensorServingInputReceiver')
class TensorServingInputReceiver(collections.namedtuple('TensorServingInputReceiver', ['features', 'receiver_tensors', 'receiver_tensors_alternatives'])):
    """A return type for a serving_input_receiver_fn.

  This is for use with models that expect a single `Tensor` or `SparseTensor`
  as an input feature, as opposed to a dict of features.

  The normal `ServingInputReceiver` always returns a feature dict, even if it
  contains only one entry, and so can be used only with models that accept such
  a dict.  For models that accept only a single raw feature, the
  `serving_input_receiver_fn` provided to `Estimator.export_saved_model()`
  should return this `TensorServingInputReceiver` instead.  See:
  https://github.com/tensorflow/tensorflow/issues/11674

  Note that the receiver_tensors and receiver_tensor_alternatives arguments
  will be automatically converted to the dict representation in either case,
  because the SavedModel format requires each input `Tensor` to have a name
  (provided by the dict key).

  Attributes:
    features: A single `Tensor` or `SparseTensor`, representing the feature to
      be passed to the model.
    receiver_tensors: A `Tensor`, `SparseTensor`, or dict of string to `Tensor`
      or `SparseTensor`, specifying input nodes where this receiver expects to
      be fed by default.  Typically, this is a single placeholder expecting
      serialized `tf.Example` protos.
    receiver_tensors_alternatives: a dict of string to additional groups of
      receiver tensors, each of which may be a `Tensor`, `SparseTensor`, or dict
      of string to `Tensor` or`SparseTensor`. These named receiver tensor
      alternatives generate additional serving signatures, which may be used to
      feed inputs at different points within the input receiver subgraph.  A
      typical usage is to allow feeding raw feature `Tensor`s *downstream* of
      the tf.parse_example() op. Defaults to None.
  """

    def __new__(cls, features, receiver_tensors, receiver_tensors_alternatives=None):
        if features is None:
            raise ValueError('features must be defined.')
        _check_tensor(features, None)
        receiver = ServingInputReceiver(features=features, receiver_tensors=receiver_tensors, receiver_tensors_alternatives=receiver_tensors_alternatives)
        return super(TensorServingInputReceiver, cls).__new__(cls, features=receiver.features[SINGLE_FEATURE_DEFAULT_NAME], receiver_tensors=receiver.receiver_tensors, receiver_tensors_alternatives=receiver.receiver_tensors_alternatives)