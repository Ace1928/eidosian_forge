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
@add_start_docstrings('T5 Model with a `language modeling` head on top.', T5_START_DOCSTRING)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model_dim = config.d_model
        self.shared = keras.layers.Embedding(config.vocab_size, config.d_model, name='shared', embeddings_initializer=get_initializer(self.config.initializer_factor))
        self.shared.load_weight_prefix = 'shared'
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name='encoder')
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFT5MainLayer(decoder_config, self.shared, name='decoder')
        if not config.tie_word_embeddings:
            lm_head_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor)
            self.lm_head = keras.layers.Dense(config.vocab_size, use_bias=False, name='lm_head', kernel_initializer=lm_head_initializer)
        self.config = config

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.get_input_embeddings()
        else:
            return tf.transpose(self.lm_head.kernel)

    def set_output_embeddings(self, value):
        if self.config.tie_word_embeddings:
            self.set_input_embeddings(value)
        else:
            lm_head_initializer = keras.initializers.RandomNormal(mean=0, stddev=self.config.initializer_factor)
            self.lm_head = keras.layers.Dense(shape_list(value)[0], use_bias=False, name='lm_head', kernel_initializer=lm_head_initializer)
            transposed_value = tf.transpose(value)
            self.lm_head.kernel = transposed_value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, labels: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFSeq2SeqLMOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = TFT5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> inputs = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="tf").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="tf").input_ids
        >>> outputs = model(inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> inputs = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(inputs)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you
        ```"""
        if head_mask is not None and decoder_head_mask is None:
            warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
            decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        decoder_outputs = self.decoder(decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, inputs_embeds=decoder_inputs_embeds, head_mask=decoder_head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
            logits = tf.matmul(sequence_output, self.shared.weights, transpose_b=True)
        else:
            logits = self.lm_head(sequence_output)
        logits = tf.cast(logits, tf.float32)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        past = decoder_outputs[1] if use_cache else None
        if not return_dict:
            if past_key_values is not None:
                decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output
        elif isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
            hidden_states = None
            attentions = None
            idx = 0
            if output_hidden_states:
                idx += 1
                hidden_states = encoder_outputs[idx]
            if output_attentions:
                idx += 1
                attentions = encoder_outputs[idx]
            encoder_outputs = TFBaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions)
        return TFSeq2SeqLMOutput(loss=loss, logits=logits, past_key_values=past, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values[1:]) if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqLMOutput(logits=output.logits, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'input_ids': None, 'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'use_cache': use_cache}

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return self._shift_right(labels)

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
        if getattr(self, 'lm_head', None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.d_model])