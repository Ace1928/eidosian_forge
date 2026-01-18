from typing import Dict
from typing import List
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import adapters
from autokeras import analysers
from autokeras import blocks
from autokeras import hyper_preprocessors as hpps_module
from autokeras import keras_layers
from autokeras import preprocessors
from autokeras.engine import io_hypermodel
from autokeras.engine import node as node_module
class ImageInput(Input):
    """Input node for image data.

    The input data should be numpy.ndarray or tf.data.Dataset. The shape of the data
    should be should be (samples, width, height) or
    (samples, width, height, channels).

    # Arguments
        name: String. The name of the input node. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: Optional[str]=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, hp, inputs=None):
        inputs = super().build(hp, inputs)
        output_node = nest.flatten(inputs)[0]
        if len(output_node.shape) == 3:
            output_node = keras_layers.ExpandLastDim()(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ImageAdapter()

    def get_analyser(self):
        return analysers.ImageAnalyser()

    def get_block(self):
        return blocks.ImageBlock()