from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
class TensorBoardVersionSelector(object):
    """Chooses between Keras v1 and v2 TensorBoard callback class."""

    def __new__(cls, *args, **kwargs):
        use_v2 = should_use_v2()
        start_cls = cls
        cls = swap_class(start_cls, callbacks.TensorBoard, callbacks_v1.TensorBoard, use_v2)
        if start_cls == callbacks_v1.TensorBoard and cls == callbacks.TensorBoard:
            return cls(*args, **kwargs)
        return super(TensorBoardVersionSelector, cls).__new__(cls)