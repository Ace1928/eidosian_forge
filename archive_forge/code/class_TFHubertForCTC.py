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
@add_start_docstrings('TFHubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).', HUBERT_START_DOCSTRING)
class TFHubertForCTC(TFHubertPreTrainedModel):

    def __init__(self, config: HubertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.hubert = TFHubertMainLayer(config, name='hubert')
        self.dropout = keras.layers.Dropout(config.final_dropout)
        self.lm_head = keras.layers.Dense(config.vocab_size, name='lm_head')
        self.output_hidden_size = config.output_hidden_size if hasattr(config, 'add_adapter') and config.add_adapter else config.hidden_size

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor.trainable = False

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, labels: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_values` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoProcessor, TFHubertForCTC
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = TFHubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> logits = model(input_values).logits
        >>> predicted_ids = tf.argmax(logits, axis=-1)

        >>> transcription = processor.decode(predicted_ids[0])

        >>> # compute loss
        >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

        >>> # Pass the transcription as text to encode labels
        >>> labels = processor(text=transcription, return_tensors="tf").input_values

        >>> loss = model(input_values, labels=labels).loss
        ```"""
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, training=training)
        logits = self.lm_head(hidden_states)
        if labels is not None:
            if tf.reduce_max(labels) >= self.config.vocab_size:
                raise ValueError(f'Label values must be <= vocab_size: {self.config.vocab_size}')
            attention_mask = attention_mask if attention_mask is not None else tf.ones_like(input_values, dtype=tf.float32)
            input_lengths = self.hubert._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, axis=-1))
            labels_mask = tf.cast(labels >= 0, tf.int32)
            target_lengths = tf.reduce_sum(labels_mask, axis=-1)
            loss = tf.nn.ctc_loss(logits=logits, labels=labels, logit_length=input_lengths, label_length=target_lengths, blank_index=self.config.pad_token_id, logits_time_major=False)
            if self.config.ctc_loss_reduction == 'sum':
                loss = tf.reduce_sum(loss)
                loss = tf.reshape(loss, (1,))
            if self.config.ctc_loss_reduction == 'mean':
                loss = tf.reduce_mean(loss)
                loss = tf.reshape(loss, (1,))
        else:
            loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFCausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'hubert', None) is not None:
            with tf.name_scope(self.hubert.name):
                self.hubert.build(None)
        if getattr(self, 'lm_head', None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.output_hidden_size])