import threading
from tensorboard.compat import tf2 as tf
def _lazy_tensor_creator_converter(value, dtype=None, name=None, as_ref=False):
    del name
    if not isinstance(value, LazyTensorCreator):
        raise RuntimeError('Expected LazyTensorCreator, got %r' % value)
    if as_ref:
        raise RuntimeError('Cannot use LazyTensorCreator to create ref tensor')
    tensor = value()
    if dtype not in (None, tensor.dtype):
        raise RuntimeError('Cannot convert LazyTensorCreator returning dtype %s to dtype %s' % (tensor.dtype, dtype))
    return tensor