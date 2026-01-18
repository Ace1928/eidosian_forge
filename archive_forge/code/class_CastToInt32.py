import tensorflow as tf
from autokeras import analysers
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import preprocessor
from autokeras.utils import data_utils
class CastToInt32(preprocessor.Preprocessor):
    """Cast the dataset shape to tf.int32."""

    def transform(self, dataset):
        return dataset.map(lambda x: tf.cast(x, tf.int32))