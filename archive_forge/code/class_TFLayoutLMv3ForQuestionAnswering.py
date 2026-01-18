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
@add_start_docstrings('\n    LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as\n    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to\n    compute `span start logits` and `span end logits`).\n    ', LAYOUTLMV3_START_DOCSTRING)
class TFLayoutLMv3ForQuestionAnswering(TFLayoutLMv3PreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ['position_ids']

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlmv3 = TFLayoutLMv3MainLayer(config, name='layoutlmv3')
        self.qa_outputs = TFLayoutLMv3ClassificationHead(config, name='qa_outputs')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLMV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, start_positions: tf.Tensor | None=None, end_positions: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, bbox: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFAutoModelForQuestionAnswering
        >>> from datasets import load_dataset
        >>> import tensorflow as tf

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = TFAutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> question = "what's his name?"
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, question, words, boxes=boxes, return_tensors="tf")
        >>> start_positions = tf.convert_to_tensor([1])
        >>> end_positions = tf.convert_to_tensor([3])

        >>> outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.layoutlmv3(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, bbox=bbox, pixel_values=pixel_values, training=training)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output, training=training)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions, 'end_position': end_positions}
            loss = self.hf_compute_loss(labels, logits=(start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layoutlmv3', None) is not None:
            with tf.name_scope(self.layoutlmv3.name):
                self.layoutlmv3.build(None)
        if getattr(self, 'qa_outputs', None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build(None)