from autokeras.engine import block as block_module
class IOHyperModel(block_module.Block):
    """A mixin class connecting the input nodes and heads with the adapters.

    This class is extended by the input nodes and the heads. The AutoModel calls the
    functions to get the corresponding adapters and pass the information back to the
    input nodes and heads.
    """

    def __init__(self, shape=None, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.data_shape = None
        self.dtype = None
        self.batch_size = None
        self.num_samples = None

    def get_analyser(self):
        """Get the corresponding Analyser.

        # Returns
            An instance of a subclass of autokeras.engine.Analyser.
        """
        raise NotImplementedError

    def get_adapter(self):
        """Get the corresponding Adapter.

        # Returns
            An instance of a subclass of autokeras.engine.Adapter.
        """
        raise NotImplementedError

    def config_from_analyser(self, analyser):
        """Load the learned information on dataset from the Analyser.

        # Arguments
            adapter: An instance of a subclass of autokeras.engine.Adapter.
        """
        self.data_shape = analyser.shape
        self.dtype = analyser.dtype
        self.batch_size = analyser.batch_size
        self.num_samples = analyser.num_samples

    def get_hyper_preprocessors(self):
        """Construct a list of HyperPreprocessors based on the learned information.

        # Returns
            A list of HyperPreprocessors for the corresponding data.
        """
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({'shape': self.shape})
        return config