import copy
import inspect
import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.layers.core.input_layer import InputLayer
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.functional import Functional
from keras.src.models.model import Model
from keras.src.saving import serialization_lib
def _maybe_rebuild(self):
    self.built = False
    self._functional = None
    if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
        input_shape = self._layers[0].batch_shape
        self.build(input_shape)