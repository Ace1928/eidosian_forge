from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
class ModelVersionSelector(object):
    """Chooses between Keras v1 and v2 Model class."""

    def __new__(cls, *args, **kwargs):
        use_v2 = should_use_v2()
        cls = swap_class(cls, training.Model, training_v1.Model, use_v2)
        return super(ModelVersionSelector, cls).__new__(cls)