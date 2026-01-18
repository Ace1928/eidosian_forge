from __future__ import annotations
import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_t5 import T5Config
@add_start_docstrings('The bare T5 Model transformer outputting raw hidden-stateswithout any specific head on top.', T5_START_DOCSTRING)
class TFT5Model(TFT5PreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.shared = keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.d_model, embeddings_initializer=keras.initializers.TruncatedNormal(self.config.initializer_factor), name='shared')
        self.shared.load_weight_prefix = 'shared'
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name='encoder')
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFT5MainLayer(decoder_config, self.shared, name='decoder')

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFSeq2SeqModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFT5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = TFT5Model.from_pretrained("google-t5/t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        if head_mask is not None and decoder_head_mask is None:
            warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=inputs_embeds, head_mask=head_mask, past_key_values=None, use_cache=False, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, inputs_embeds=decoder_inputs_embeds, head_mask=decoder_head_mask, encoder_head_mask=head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        past = decoder_outputs[1] if use_cache else None
        if not return_dict:
            if past_key_values is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            return decoder_outputs + encoder_outputs
        return TFSeq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=past, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope(self.shared.load_weight_prefix + '/' + self.shared.name + '/'):
            self.shared.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'decoder', None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)