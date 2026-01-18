from autokeras.engine import named_hypermodel
class HyperPreprocessor(named_hypermodel.NamedHyperModel):
    """Input data preprocessor search space.

    This class defines the search space for a Preprocessor.
    """

    def build(self, hp, dataset):
        """Build the `tf.data` input preprocessor.

        # Arguments
            hp: `HyperParameters` instance. The hyperparameters for building the
                a Preprocessor.
            dataset: tf.data.Dataset.

        # Returns
            an instance of Preprocessor.
        """
        raise NotImplementedError