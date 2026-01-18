from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
class RNNAttributes(SerializedAttributes.with_attributes('RNNAttributes', checkpointable_objects=['states'], copy_from=[LayerAttributes])):
    """RNN checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from LayerAttributes (including CommonEndpoints)
    states: List of state variables
  """