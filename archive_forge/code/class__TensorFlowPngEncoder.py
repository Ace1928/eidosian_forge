import numpy as np
from tensorboard.util import op_evaluator
class _TensorFlowPngEncoder(op_evaluator.PersistentOpEvaluator):
    """Encode an image to PNG.

    This function is thread-safe, and has high performance when run in
    parallel. See `encode_png_benchmark.py` for details.

    Arguments:
      image: A numpy array of shape `[height, width, channels]`, where
        `channels` is 1, 3, or 4, and of dtype uint8.

    Returns:
      A bytestring with PNG-encoded data.
    """

    def __init__(self):
        super().__init__()
        self._image_placeholder = None
        self._encode_op = None

    def initialize_graph(self):
        import tensorflow.compat.v1 as tf
        self._image_placeholder = tf.placeholder(dtype=tf.uint8, name='image_to_encode')
        self._encode_op = tf.image.encode_png(self._image_placeholder)

    def run(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("'image' must be a numpy array: %r" % image)
        if image.dtype != np.uint8:
            raise ValueError("'image' dtype must be uint8, but is %r" % image.dtype)
        return self._encode_op.eval(feed_dict={self._image_placeholder: image})