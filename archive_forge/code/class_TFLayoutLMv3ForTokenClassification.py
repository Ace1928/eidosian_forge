from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
@add_start_docstrings('\n    LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.\n    for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),\n    [SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and\n    [Kleister-NDA](https://github.com/applicaai/kleister-nda).\n    ', LAYOUTLMV3_START_DOCSTRING)
class TFLayoutLMv3ForTokenClassification(TFLayoutLMv3PreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['position_ids']

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name='layoutlmv3')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')
        if config.num_labels < 10:
            self.classifier = keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        else:
            self.classifier = TFLayoutLMv3ClassificationHead(config, name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: tf.Tensor | None=None, bbox: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, pixel_values: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="tf")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.layoutlmv3(input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, pixel_values=pixel_values, training=training)
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]
        seq_length = input_shape[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layoutlmv3', None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])