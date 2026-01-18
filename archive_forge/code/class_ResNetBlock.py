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
class ResNetBlock(KerasApplicationBlock):
    """Block for ResNet.

    # Arguments
        version: String. 'v1', 'v2'. The type of ResNet to use.
            If left unspecified, it will be tuned automatically.
        pretrained: Boolean. Whether to use ImageNet pretrained weights.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, version: Optional[str]=None, pretrained: Optional[bool]=None, **kwargs):
        if version is None:
            models = {**RESNET_V1, **RESNET_V2}
        elif version == 'v1':
            models = RESNET_V1
        elif version == 'v2':
            models = RESNET_V2
        else:
            raise ValueError('Expect version to be "v1", or "v2", but got {version}.'.format(version=version))
        super().__init__(pretrained=pretrained, models=models, min_size=32, **kwargs)
        self.version = version

    def get_config(self):
        config = super().get_config()
        config.update({'version': self.version})
        return config