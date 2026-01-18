from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
class InputLayerSavedModelSaver(base_serialization.SavedModelSaver):
    """InputLayer serialization."""

    @property
    def object_identifier(self):
        return constants.INPUT_LAYER_IDENTIFIER

    @property
    def python_properties(self):
        return dict(class_name=type(self.obj).__name__, name=self.obj.name, dtype=self.obj.dtype, sparse=self.obj.sparse, ragged=self.obj.ragged, batch_input_shape=self.obj._batch_input_shape, config=self.obj.get_config())

    def objects_to_serialize(self, serialization_cache):
        return {}

    def functions_to_serialize(self, serialization_cache):
        return {}