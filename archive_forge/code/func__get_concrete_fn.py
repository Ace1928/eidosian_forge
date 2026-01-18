import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def _get_concrete_fn(self, endpoint):
    """Workaround for some SavedModel quirks."""
    if endpoint in self._endpoint_signatures:
        return getattr(self, endpoint)
    else:
        traces = getattr(self, endpoint)._trackable_children('saved_model')
        return list(traces.values())[0]