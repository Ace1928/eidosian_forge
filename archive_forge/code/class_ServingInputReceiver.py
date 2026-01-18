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
@estimator_export('estimator.export.ServingInputReceiver')
class ServingInputReceiver(collections.namedtuple('ServingInputReceiver', ['features', 'receiver_tensors', 'receiver_tensors_alternatives'])):
    """A return type for a serving_input_receiver_fn.

  Attributes:
    features: A `Tensor`, `SparseTensor`, or dict of string or int to `Tensor`
      or `SparseTensor`, specifying the features to be passed to the model.
      Note: if `features` passed is not a dict, it will be wrapped in a dict
        with a single entry, using 'feature' as the key.  Consequently, the
        model
      must accept a feature dict of the form {'feature': tensor}.  You may use
        `TensorServingInputReceiver` if you want the tensor to be passed as is.
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
        features = wrap_and_check_input_tensors(features, 'feature', allow_int_keys=True)
        receiver_tensors = wrap_and_check_input_tensors(receiver_tensors, 'receiver_tensor')
        if receiver_tensors_alternatives is not None:
            if not isinstance(receiver_tensors_alternatives, dict):
                raise ValueError('receiver_tensors_alternatives must be a dict: {}.'.format(receiver_tensors_alternatives))
            for alternative_name, receiver_tensors_alt in six.iteritems(receiver_tensors_alternatives):
                receiver_tensors_alternatives[alternative_name] = wrap_and_check_input_tensors(receiver_tensors_alt, 'receiver_tensors_alternative')
        return super(ServingInputReceiver, cls).__new__(cls, features=features, receiver_tensors=receiver_tensors, receiver_tensors_alternatives=receiver_tensors_alternatives)