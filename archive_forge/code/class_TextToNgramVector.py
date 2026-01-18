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
class TextToNgramVector(block_module.Block):
    """Convert raw texts to n-gram vectors.

    # Arguments
        max_tokens: Int. The maximum size of the vocabulary. Defaults to 20000.
        ngrams: Int or tuple of ints. Passing an integer will create ngrams up to
            that integer, and passing a tuple of integers will create ngrams for the
            specified values in the tuple. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(self, max_tokens: int=20000, ngrams: Union[int, Tuple[int], None]=None, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.ngrams = ngrams

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        if self.ngrams is not None:
            ngrams = self.ngrams
        else:
            ngrams = hp.Int('ngrams', min_value=1, max_value=2, default=2)
        return layers.TextVectorization(max_tokens=self.max_tokens, ngrams=ngrams, output_mode='tf-idf', pad_to_max_tokens=True)(input_node)

    def get_config(self):
        config = super().get_config()
        config.update({'max_tokens': self.max_tokens, 'ngrams': self.ngrams})
        return config