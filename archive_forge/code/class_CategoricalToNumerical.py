from typing import Optional
from typing import Tuple
from typing import Union
from keras_tuner.engine import hyperparameters
from tensorflow import nest
from tensorflow.keras import layers
from autokeras import analysers
from autokeras import keras_layers
from autokeras.engine import block as block_module
from autokeras.utils import io_utils
from autokeras.utils import utils
class CategoricalToNumerical(block_module.Block):
    """Encode the categorical features to numerical features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.column_types = None
        self.column_names = None

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        encoding = []
        for column_name in self.column_names:
            column_type = self.column_types[column_name]
            if column_type == analysers.CATEGORICAL:
                encoding.append(keras_layers.INT)
            else:
                encoding.append(keras_layers.NONE)
        return keras_layers.MultiCategoryEncoding(encoding)(input_node)

    @classmethod
    def from_config(cls, config):
        column_types = config.pop('column_types')
        column_names = config.pop('column_names')
        instance = cls(**config)
        instance.column_types = column_types
        instance.column_names = column_names
        return instance

    def get_config(self):
        config = super().get_config()
        config.update({'column_types': self.column_types, 'column_names': self.column_names})
        return config