from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
class LayerVersionSelector(object):
    """Chooses between Keras v1 and v2 Layer class."""

    def __new__(cls, *args, **kwargs):
        use_v2 = should_use_v2()
        cls = swap_class(cls, base_layer.Layer, base_layer_v1.Layer, use_v2)
        return super(LayerVersionSelector, cls).__new__(cls)