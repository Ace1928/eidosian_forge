import numpy as np
from autokeras.engine import preprocessor
class SoftmaxPostprocessor(PostProcessor):
    """Postprocessor for softmax outputs."""

    def postprocess(self, data):
        """Transform probabilities to zeros and ones.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The zeros and ones predictions.
        """
        idx = np.argmax(data, axis=-1)
        data = np.zeros(data.shape)
        data[np.arange(data.shape[0]), idx] = 1
        return data