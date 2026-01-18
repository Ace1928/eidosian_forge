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
class ConvBlock(block_module.Block):
    """Block for vanilla ConvNets.

    # Arguments
        kernel_size: Int or keras_tuner.engine.hyperparameters.Choice.
            The size of the kernel.
            If left unspecified, it will be tuned automatically.
        num_blocks: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left unspecified,
            it will be tuned automatically.
        num_layers: Int or hyperparameters.Choice.
            The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        filters: Int or keras_tuner.engine.hyperparameters.Choice. The number of
            filters in the convolutional layers. If left unspecified, it will
            be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If left
            unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float or kerastuner.engine.hyperparameters.
            Choice range Between 0 and 1.
            The dropout rate after convolutional layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, kernel_size: Optional[Union[int, hyperparameters.Choice]]=None, num_blocks: Optional[Union[int, hyperparameters.Choice]]=None, num_layers: Optional[Union[int, hyperparameters.Choice]]=None, filters: Optional[Union[int, hyperparameters.Choice]]=None, max_pooling: Optional[bool]=None, separable: Optional[bool]=None, dropout: Optional[Union[float, hyperparameters.Choice]]=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = utils.get_hyperparameter(kernel_size, hyperparameters.Choice('kernel_size', [3, 5, 7], default=3), int)
        self.num_blocks = utils.get_hyperparameter(num_blocks, hyperparameters.Choice('num_blocks', [1, 2, 3], default=2), int)
        self.num_layers = utils.get_hyperparameter(num_layers, hyperparameters.Choice('num_layers', [1, 2], default=2), int)
        self.filters = utils.get_hyperparameter(filters, hyperparameters.Choice('filters', [16, 32, 64, 128, 256, 512], default=32), int)
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout = utils.get_hyperparameter(dropout, hyperparameters.Choice('dropout', [0.0, 0.25, 0.5], default=0.0), float)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': io_utils.serialize_block_arg(self.kernel_size), 'num_blocks': io_utils.serialize_block_arg(self.num_blocks), 'num_layers': io_utils.serialize_block_arg(self.num_layers), 'filters': io_utils.serialize_block_arg(self.filters), 'max_pooling': self.max_pooling, 'separable': self.separable, 'dropout': io_utils.serialize_block_arg(self.dropout)})
        return config

    @classmethod
    def from_config(cls, config):
        config['kernel_size'] = io_utils.deserialize_block_arg(config['kernel_size'])
        config['num_blocks'] = io_utils.deserialize_block_arg(config['num_blocks'])
        config['num_layers'] = io_utils.deserialize_block_arg(config['num_layers'])
        config['filters'] = io_utils.deserialize_block_arg(config['filters'])
        config['dropout'] = io_utils.deserialize_block_arg(config['dropout'])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        kernel_size = utils.add_to_hp(self.kernel_size, hp)
        separable = self.separable
        if separable is None:
            separable = hp.Boolean('separable', default=False)
        if separable:
            conv = layer_utils.get_sep_conv(input_node.shape)
        else:
            conv = layer_utils.get_conv(input_node.shape)
        max_pooling = self.max_pooling
        if max_pooling is None:
            max_pooling = hp.Boolean('max_pooling', default=True)
        pool = layer_utils.get_max_pooling(input_node.shape)
        for i in range(utils.add_to_hp(self.num_blocks, hp)):
            for j in range(utils.add_to_hp(self.num_layers, hp)):
                output_node = conv(utils.add_to_hp(self.filters, hp, 'filters_{i}_{j}'.format(i=i, j=j)), kernel_size, padding=self._get_padding(kernel_size, output_node), activation='relu')(output_node)
            if max_pooling:
                output_node = pool(kernel_size - 1, padding=self._get_padding(kernel_size - 1, output_node))(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        if all((kernel_size * 2 <= length for length in output_node.shape[1:-1])):
            return 'valid'
        return 'same'