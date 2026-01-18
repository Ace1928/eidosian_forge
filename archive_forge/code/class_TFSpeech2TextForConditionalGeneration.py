from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
@add_start_docstrings('The Speech2Text Model with a language modeling head. Can be used for summarization.', SPEECH_TO_TEXT_START_DOCSTRING)
class TFSpeech2TextForConditionalGeneration(TFSpeech2TextPreTrainedModel, TFCausalLanguageModelingLoss):

    def __init__(self, config: Speech2TextConfig):
        super().__init__(config)
        self.model = TFSpeech2TextMainLayer(config, name='model')
        self.lm_head = keras.layers.Dense(self.config.vocab_size, use_bias=False, name='lm_head')
        self.supports_xla_generation = False
        self.config = config

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> tf.Variable:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_features: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, labels: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False, **kwargs) -> Union[Tuple, TFSeq2SeqLMOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> model = TFSpeech2TextForConditionalGeneration.from_pretrained(
        ...     "facebook/s2t-small-librispeech-asr", from_pt=True
        ... )
        >>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)
        >>> ds.set_format(type="tf")

        >>> input_features = processor(
        ...     ds["speech"][0], sampling_rate=16000, return_tensors="tf"
        ... ).input_features  # Batch size 1
        >>> generated_ids = model.generate(input_features)

        >>> transcription = processor.batch_decode(generated_ids)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.model(input_features=input_features, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        lm_logits = self.lm_head(outputs[0])
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return TFSeq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqLMOutput(logits=output.logits, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_features': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'model', None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, 'lm_head', None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.d_model])

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == 'lm_head.weight':
            return (tf_weight, 'model.decoder.embed_tokens.weight')
        else:
            return (tf_weight,)