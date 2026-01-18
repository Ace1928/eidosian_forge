from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
@add_start_docstrings('\n    The Transfo XL Model transformer with a sequence classification head on top (linear layer).\n\n    [`TFTransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal\n    models (e.g. GPT-1,GPT-2) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLForSequenceClassification(TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.score = keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.init_range), name='score', use_bias=False)
        self.transformer = TFTransfoXLMainLayer(config, name='transformer')

    def get_output_embeddings(self):
        logger.warning('Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed in transformers v4.32.')
        return self.transformer.word_emb

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFTransfoXLSequenceClassifierOutputWithPast]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        transformer_outputs = self.transformer(input_ids=input_ids, mems=mems, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        in_logits = None
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1) - 1
            sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
            in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`')
        loss = None
        if labels is not None:
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            assert self.config.pad_token_id is not None or batch_size == 1, 'Cannot handle batch sizes > 1 if no padding token is defined.'
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))
        pooled_logits = in_logits if in_logits is not None else logits
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFTransfoXLSequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)