import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import preprocessors as preprocessors_module
from autokeras.engine import hyper_preprocessor as hpps_module
from autokeras.engine import preprocessor as pps_module
from autokeras.utils import data_utils
from autokeras.utils import io_utils
class HyperPipeline(hpps_module.HyperPreprocessor):
    """A search space consists of HyperPreprocessors.

    # Arguments
        inputs: a list of lists of HyperPreprocessors.
        outputs: a list of lists of HyperPreprocessors.
    """

    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def _build_preprocessors(hp, hpps_lists, dataset):
        sources = data_utils.unzip_dataset(dataset)
        preprocessors_list = []
        for source, hpps_list in zip(sources, hpps_lists):
            data = source
            preprocessors = []
            for hyper_preprocessor in hpps_list:
                preprocessor = hyper_preprocessor.build(hp, data)
                preprocessor.fit(data)
                data = preprocessor.transform(data)
                preprocessors.append(preprocessor)
            preprocessors_list.append(preprocessors)
        return preprocessors_list

    def build(self, hp, dataset):
        """Build a Pipeline by Hyperparameters.

        # Arguments
            hp: Hyperparameters.
            dataset: tf.data.Dataset.

        # Returns
            An instance of Pipeline.
        """
        x = dataset.map(lambda x, y: x)
        y = dataset.map(lambda x, y: y)
        return Pipeline(inputs=self._build_preprocessors(hp, self.inputs, x), outputs=self._build_preprocessors(hp, self.outputs, y))