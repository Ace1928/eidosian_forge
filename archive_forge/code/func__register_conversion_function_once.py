import threading
from tensorboard.compat import tf2 as tf
def _register_conversion_function_once():
    """Performs one-time registration of `_lazy_tensor_creator_converter`.

    This helper can be invoked multiple times but only registers the conversion
    function on the first invocation, making it suitable for calling when
    constructing a LazyTensorCreator.

    Deferring the registration is necessary because doing it at at module import
    time would trigger the lazy TensorFlow import to resolve, and that in turn
    would break the delicate `tf.summary` import cycle avoidance scheme.
    """
    global _conversion_registered
    if not _conversion_registered:
        with _conversion_registered_lock:
            if not _conversion_registered:
                _conversion_registered = True
                tf.register_tensor_conversion_function(base_type=LazyTensorCreator, conversion_func=_lazy_tensor_creator_converter, priority=0)