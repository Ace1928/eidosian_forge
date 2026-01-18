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
class XceptionBlock(KerasApplicationBlock):
    """Block for XceptionNet.

    An Xception structure, used for specifying your model with specific datasets.

    The original Xception architecture is from
    [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357).
    The data first goes through the entry flow, then through the middle flow which
    is repeated eight times, and finally through the exit flow.

    This XceptionBlock returns a similar architecture as Xception except without
    the last (optional) fully connected layer(s) and logistic regression.
    The size of this architecture could be decided by `HyperParameters`, to get an
    architecture with a half, an identical, or a double size of the original one.

    # Arguments
        pretrained: Boolean. Whether to use ImageNet pretrained weights.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, pretrained: Optional[bool]=None, **kwargs):
        super().__init__(pretrained=pretrained, models={'xception': applications.Xception}, min_size=71, **kwargs)