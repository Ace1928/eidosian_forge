from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import base_serialization
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
class IndexLookupLayerSavedModelSaver(LayerSavedModelSaver):
    """Index lookup layer serialization."""

    @property
    def python_properties(self):
        metadata = self._python_properties_internal()
        if metadata['config'].get('has_static_table', False):
            metadata['config']['vocabulary'] = None
        return metadata