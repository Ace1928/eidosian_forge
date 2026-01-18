from autokeras.engine import serializable
class TargetPreprocessor(Preprocessor):
    """Preprocessor for target data."""

    def postprocess(self, dataset):
        """Postprocess the output of the Keras model.

        # Arguments
            dataset: numpy.ndarray. The corresponding output of the model.

        # Returns
            numpy.ndarray. The postprocessed data.
        """
        raise NotImplementedError