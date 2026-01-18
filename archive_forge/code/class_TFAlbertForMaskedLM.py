from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_albert import AlbertConfig
@add_start_docstrings('Albert Model with a `language modeling` head on top.', ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'predictions.decoder.weight']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name='albert')
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name='predictions')

    def get_lm_head(self) -> keras.layers.Layer:
        return self.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        >>> model = TFAlbertForMaskedLM.from_pretrained("albert/albert-base-v2")

        >>> # add mask_token
        >>> inputs = tokenizer(f"The capital of [MASK] is Paris.", return_tensors="tf")
        >>> logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]
        >>> predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        'france'
        ```

        ```python
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
        >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
        >>> outputs = model(**inputs, labels=labels)
        >>> round(float(outputs.loss), 2)
        0.81
        ```
        """
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.predictions(hidden_states=sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'albert', None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, 'predictions', None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)