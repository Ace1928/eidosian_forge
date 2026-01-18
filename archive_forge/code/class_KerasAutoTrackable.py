import tensorflow as tf
from keras.src.utils import tracking
class KerasAutoTrackable(tf.__internal__.tracking.AutoTrackable):
    """Manages dependencies on other objects with Keras tracking.

    Similar to TF AutoTrackable, but disabling tracking is based
    on tracking within Keras.

    This serves as an interface between Keras tracking and TF tracking.
    """

    def __setattr__(self, name, value):
        """Support self.foo = trackable syntax."""
        try:
            if getattr(self, name) is value:
                return
        except AttributeError:
            pass
        if getattr(self, '_self_setattr_tracking', True):
            value = sticky_attribute_assignment(trackable=self, value=value, name=name)
        super().__setattr__(name, value)