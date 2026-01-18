from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
class CommonEndpoints(SerializedAttributes.with_attributes('CommonEndpoints', checkpointable_objects=['variables', 'trainable_variables', 'regularization_losses'], functions=['__call__', 'call_and_return_all_conditional_losses', '_default_save_signature'])):
    """Common endpoints shared by all models loadable by Keras.

  List of all attributes:
    variables: List of all variables in the model and its sublayers.
    trainable_variables: List of all trainable variables in the model and its
      sublayers.
    regularization_losses: List of all unconditional losses (losses not
      dependent on the inputs) in the model and its sublayers.
    __call__: Function that takes inputs and returns the outputs of the model
      call function.
    call_and_return_all_conditional_losses: Function that returns a tuple of
      (call function outputs, list of all losses that depend on the inputs).
    _default_save_signature: Traced model call function. This is only included
      if the top level exported object is a Keras model.
  """