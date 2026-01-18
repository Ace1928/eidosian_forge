from typing import Optional
from typing import Union
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import applications
from tensorflow.keras import layers
from autokeras import keras_layers
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
from autokeras.utils import io_utils
from autokeras.utils import layer_utils
from autokeras.utils import utils
class DenseBlock(block_module.Block):
    """Block for Dense layers.

    # Arguments
        num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of Dense layers in the block.
            If left unspecified, it will be tuned automatically.
        num_units: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of units in each dense layer.
            If left unspecified, it will be tuned automatically.
        use_bn: Boolean. Whether to use BatchNormalization layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float or keras_tuner.engine.hyperparameters.Choice.
            The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, num_layers: Optional[Union[int, hyperparameters.Choice]]=None, num_units: Optional[Union[int, hyperparameters.Choice]]=None, use_batchnorm: Optional[bool]=None, dropout: Optional[Union[float, hyperparameters.Choice]]=None, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = utils.get_hyperparameter(num_layers, hyperparameters.Choice('num_layers', [1, 2, 3], default=2), int)
        self.num_units = utils.get_hyperparameter(num_units, hyperparameters.Choice('num_units', [16, 32, 64, 128, 256, 512, 1024], default=32), int)
        self.use_batchnorm = use_batchnorm
        self.dropout = utils.get_hyperparameter(dropout, hyperparameters.Choice('dropout', [0.0, 0.25, 0.5], default=0.0), float)

    def get_config(self):
        config = super().get_config()
        config.update({'num_layers': io_utils.serialize_block_arg(self.num_layers), 'num_units': io_utils.serialize_block_arg(self.num_units), 'use_batchnorm': self.use_batchnorm, 'dropout': io_utils.serialize_block_arg(self.dropout)})
        return config

    @classmethod
    def from_config(cls, config):
        config['num_layers'] = io_utils.deserialize_block_arg(config['num_layers'])
        config['num_units'] = io_utils.deserialize_block_arg(config['num_units'])
        config['dropout'] = io_utils.deserialize_block_arg(config['dropout'])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = reduction.Flatten().build(hp, output_node)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean('use_batchnorm', default=False)
        for i in range(utils.add_to_hp(self.num_layers, hp)):
            units = utils.add_to_hp(self.num_units, hp, 'units_{i}'.format(i=i))
            output_node = layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            output_node = layers.ReLU()(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(output_node)
        return output_node