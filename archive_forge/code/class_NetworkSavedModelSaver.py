from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import model_serialization
class NetworkSavedModelSaver(model_serialization.ModelSavedModelSaver):
    """Network serialization."""

    @property
    def object_identifier(self):
        return constants.NETWORK_IDENTIFIER