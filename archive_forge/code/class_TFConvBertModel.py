from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_convbert import ConvBertConfig
@add_start_docstrings('The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.', CONVBERT_START_DOCSTRING)
class TFConvBertModel(TFConvBertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.convbert = TFConvBertMainLayer(config, name='convbert')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: Optional[Union[np.array, tf.Tensor]]=None, token_type_ids: Optional[Union[np.array, tf.Tensor]]=None, position_ids: Optional[Union[np.array, tf.Tensor]]=None, head_mask: Optional[Union[np.array, tf.Tensor]]=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        outputs = self.convbert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convbert', None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)