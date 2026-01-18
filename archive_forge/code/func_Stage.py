import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def Stage(block_type, depth, group_width, filters_in, filters_out, name=None):
    """Implementation of Stage in RegNet.

    Args:
      block_type: must be one of "X", "Y", "Z"
      depth: depth of stage, number of blocks to use
      group_width: group width of all blocks in  this stage
      filters_in: input filters to this stage
      filters_out: output filters from this stage
      name: name prefix

    Returns:
      Output tensor of Stage
    """
    if name is None:
        name = str(backend.get_uid('stage'))

    def apply(inputs):
        x = inputs
        if block_type == 'X':
            x = XBlock(filters_in, filters_out, group_width, stride=2, name=f'{name}_XBlock_0')(x)
            for i in range(1, depth):
                x = XBlock(filters_out, filters_out, group_width, name=f'{name}_XBlock_{i}')(x)
        elif block_type == 'Y':
            x = YBlock(filters_in, filters_out, group_width, stride=2, name=name + '_YBlock_0')(x)
            for i in range(1, depth):
                x = YBlock(filters_out, filters_out, group_width, name=f'{name}_YBlock_{i}')(x)
        elif block_type == 'Z':
            x = ZBlock(filters_in, filters_out, group_width, stride=2, name=f'{name}_ZBlock_0')(x)
            for i in range(1, depth):
                x = ZBlock(filters_out, filters_out, group_width, name=f'{name}_ZBlock_{i}')(x)
        else:
            raise NotImplementedError(f'Block type `{block_type}` not recognized.block_type must be one of (`X`, `Y`, `Z`). ')
        return x
    return apply