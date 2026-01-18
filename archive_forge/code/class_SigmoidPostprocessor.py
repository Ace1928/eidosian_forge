import numpy as np
from autokeras.engine import preprocessor
class SigmoidPostprocessor(PostProcessor):
    """Postprocessor for sigmoid outputs."""

    def postprocess(self, data):
        """Transform probabilities to zeros and ones.

        # Arguments
            data: numpy.ndarray. The output probabilities of the classification head.

        # Returns
            numpy.ndarray. The zeros and ones predictions.
        """
        data[data < 0.5] = 0
        data[data > 0.5] = 1
        return data