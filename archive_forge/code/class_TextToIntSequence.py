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
class TextToIntSequence(block_module.Block):
    """Convert raw texts to sequences of word indices.

    # Arguments
        output_sequence_length: Int. The maximum length of a sentence. If
            unspecified, it would be tuned automatically.
        max_tokens: Int. The maximum size of the vocabulary. Defaults to 20000.
    """

    def __init__(self, output_sequence_length: Optional[int]=None, max_tokens: int=20000, **kwargs):
        super().__init__(**kwargs)
        self.output_sequence_length = output_sequence_length
        self.max_tokens = max_tokens

    def get_config(self):
        config = super().get_config()
        config.update({'output_sequence_length': self.output_sequence_length, 'max_tokens': self.max_tokens})
        return config

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        if self.output_sequence_length is not None:
            output_sequence_length = self.output_sequence_length
        else:
            output_sequence_length = hp.Choice('output_sequence_length', [64, 128, 256, 512], default=64)
        output_node = layers.TextVectorization(max_tokens=self.max_tokens, output_mode='int', output_sequence_length=output_sequence_length)(input_node)
        return output_node