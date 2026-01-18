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
class BertBlock(block_module.Block):
    """Block for Pre-trained BERT.
    The input should be sequence of sentences. The implementation is derived from
    this [example](https://www.tensorflow.org/official_models/fine_tuning_bert)

    # Example
    ```python
        # Using the Transformer Block with AutoModel.
        import autokeras as ak
        from autokeras import BertBlock
        from tensorflow.keras import losses

        input_node = ak.TextInput()
        output_node = BertBlock(max_sequence_length=128)(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
    ```
    # Arguments
        max_sequence_length: Int or keras_tuner.engine.hyperparameters.Choice.
            The maximum length of a sequence that is used to train the model.
    """

    def __init__(self, max_sequence_length: Optional[Union[int, hyperparameters.Choice]]=None, **kwargs):
        super().__init__(**kwargs)
        self.max_sequence_length = utils.get_hyperparameter(max_sequence_length, hyperparameters.Choice('max_sequence_length', [128, 256, 512], default=128), int)

    def get_config(self):
        config = super().get_config()
        config.update({'max_sequence_length': io_utils.serialize_block_arg(self.max_sequence_length)})
        return config

    @classmethod
    def from_config(cls, config):
        config['max_sequence_length'] = io_utils.deserialize_block_arg(config['max_sequence_length'])
        return cls(**config)

    def build(self, hp, inputs=None):
        input_tensor = nest.flatten(inputs)[0]
        tokenizer_layer = keras_layers.BertTokenizer(max_sequence_length=utils.add_to_hp(self.max_sequence_length, hp))
        output_node = tokenizer_layer(input_tensor)
        bert_encoder = keras_layers.BertEncoder()
        output_node = bert_encoder(output_node)
        bert_encoder.load_pretrained_weights()
        return output_node