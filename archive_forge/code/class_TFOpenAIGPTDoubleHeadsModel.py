from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_openai import OpenAIGPTConfig
@add_start_docstrings('\n    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for\n    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the\n    input embeddings, the classification head takes as input the input of a specified classification token index in the\n    input sequence).\n    ', OPENAI_GPT_START_DOCSTRING)
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config.num_labels = 1
        self.transformer = TFOpenAIGPTMainLayer(config, name='transformer')
        self.multiple_choice_head = TFSequenceSummary(config, initializer_range=config.initializer_range, name='multiple_choice_head')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFOpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, mc_token_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFOpenAIGPTDoubleHeadsModelOutput]:
        """
        mc_token_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
            1]`.

        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFOpenAIGPTDoubleHeadsModel

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
        >>> model = TFOpenAIGPTDoubleHeadsModel.from_pretrained("openai-community/openai-gpt")

        >>> # Add a [CLS] to the vocabulary (we should train it also!)
        >>> tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        >>> model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        >>> print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        >>> encoding = tokenizer(choices, return_tensors="tf")
        >>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
        >>> inputs["mc_token_ids"] = tf.constant(
        ...     [inputs["input_ids"].shape[-1] - 1, inputs["input_ids"].shape[-1] - 1]
        ... )[
        ...     None, :
        ... ]  # Batch size 1
        >>> outputs = model(inputs)
        >>> lm_prediction_scores, mc_prediction_scores = outputs[:2]
        ```"""
        if input_ids is not None:
            input_shapes = shape_list(input_ids)
        else:
            input_shapes = shape_list(inputs_embeds)[:-1]
        seq_length = input_shapes[-1]
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        transformer_outputs = self.transformer(flat_input_ids, flat_attention_mask, flat_token_type_ids, flat_position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        hidden_states = tf.reshape(hidden_states, input_shapes + shape_list(hidden_states)[-1:])
        if return_dict and output_hidden_states:
            all_hidden_states = transformer_outputs.hidden_states[:-1] + (hidden_states,)
        else:
            all_hidden_states = None
        lm_logits = self.transformer.tokens_embed(hidden_states, mode='linear')
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids, training=training)
        mc_logits = tf.squeeze(mc_logits, axis=-1)
        if not return_dict:
            return (lm_logits, mc_logits) + transformer_outputs[1:]
        return TFOpenAIGPTDoubleHeadsModelOutput(logits=lm_logits, mc_logits=mc_logits, hidden_states=all_hidden_states, attentions=transformer_outputs.attentions)

    @property
    def input_signature(self):
        return {'input_ids': tf.TensorSpec((None, None, None), tf.int32, name='input_ids'), 'attention_mask': tf.TensorSpec((None, None, None), tf.int32, name='attention_mask'), 'mc_token_ids': tf.TensorSpec((None, None), tf.int32, name='token_type_ids')}

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, 'multiple_choice_head', None) is not None:
            with tf.name_scope(self.multiple_choice_head.name):
                self.multiple_choice_head.build(None)