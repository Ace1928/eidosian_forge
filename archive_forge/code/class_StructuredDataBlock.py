from typing import Optional
from tensorflow import nest
from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
class StructuredDataBlock(block_module.Block):
    """Block for structured data.

    # Arguments
        categorical_encoding: Boolean. Whether to use the CategoricalToNumerical to
            encode the categorical features to numerical features. Defaults to True.
        normalize: Boolean. Whether to normalize the features.
            If unspecified, it will be tuned automatically.
        seed: Int. Random seed.
    """

    def __init__(self, categorical_encoding: bool=True, normalize: Optional[bool]=None, seed: Optional[int]=None, **kwargs):
        super().__init__(**kwargs)
        self.categorical_encoding = categorical_encoding
        self.normalize = normalize
        self.seed = seed
        self.column_types = None
        self.column_names = None

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
        config.update({'categorical_encoding': self.categorical_encoding, 'normalize': self.normalize, 'seed': self.seed, 'column_types': self.column_types, 'column_names': self.column_names})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        if self.categorical_encoding:
            block = preprocessing.CategoricalToNumerical()
            block.column_types = self.column_types
            block.column_names = self.column_names
            output_node = block.build(hp, output_node)
        if self.normalize is None and hp.Boolean(NORMALIZE):
            with hp.conditional_scope(NORMALIZE, [True]):
                output_node = preprocessing.Normalization().build(hp, output_node)
        elif self.normalize:
            output_node = preprocessing.Normalization().build(hp, output_node)
        output_node = basic.DenseBlock().build(hp, output_node)
        return output_node