from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roformer import RoFormerConfig
@add_start_docstrings('RoFormer Model with a `language modeling` head on top.', ROFORMER_START_DOCSTRING)
class TFRoFormerForMaskedLM(TFRoFormerPreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if config.is_decoder:
            logger.warning('If you want to use `TFRoFormerForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention.')
        self.roformer = TFRoFormerMainLayer(config, name='roformer')
        self.mlm = TFRoFormerMLMHead(config, input_embeddings=self.roformer.embeddings, name='mlm___cls')

    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.roformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'roformer', None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        if getattr(self, 'mlm', None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)