from __future__ import annotations
import warnings
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_hubert import HubertConfig
class TFHubertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = HubertConfig
    base_model_prefix = 'hubert'
    main_input_name = 'input_values'

    @property
    def input_signature(self):
        return {'input_values': tf.TensorSpec((None, 16000), tf.float32, name='input_values'), 'attention_mask': tf.TensorSpec((None, None), tf.int32, name='attention_mask'), 'token_type_ids': tf.TensorSpec((None, None), tf.int32, name='token_type_ids')}

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        logger.warning(f'\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish to train/fine-tune this model, you need a GPU or a TPU')