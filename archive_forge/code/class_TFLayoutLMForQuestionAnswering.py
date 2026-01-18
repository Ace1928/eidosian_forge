from __future__ import annotations
import math
import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig
@add_start_docstrings('\n    LayoutLM Model with a span classification head on top for extractive question-answering tasks such as\n    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span\n    start logits` and `span end logits`).\n    ', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForQuestionAnswering(TFLayoutLMPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'mlm___cls', 'nsp___cls', 'cls.predictions', 'cls.seq_relationship']

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name='layoutlm')
        self.qa_outputs = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        """
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFLayoutLMForQuestionAnswering
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
        >>> model = TFLayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")

        >>> dataset = load_dataset("nielsr/funsd", split="train")
        >>> example = dataset[0]
        >>> question = "what's his name?"
        >>> words = example["words"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(
        ...     question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="tf"
        ... )
        >>> bbox = []
        >>> for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
        ...     if s == 1:
        ...         bbox.append(boxes[w])
        ...     elif i == tokenizer.sep_token_id:
        ...         bbox.append([1000] * 4)
        ...     else:
        ...         bbox.append([0] * 4)
        >>> encoding["bbox"] = tf.convert_to_tensor([bbox])

        >>> word_ids = encoding.word_ids(0)
        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
        >>> start, end = word_ids[tf.math.argmax(start_scores, -1)[0]], word_ids[tf.math.argmax(end_scores, -1)[0]]
        >>> print(" ".join(words[start : end + 1]))
        M. Hamann P. Harper, P. Martinez
        ```"""
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.qa_outputs(inputs=sequence_output)
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layoutlm', None) is not None:
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
        if getattr(self, 'qa_outputs', None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])