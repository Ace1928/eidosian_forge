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
@add_start_docstrings('\n    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for\n    Named-Entity-Recognition (NER) tasks.\n    ', LAYOUTLM_START_DOCSTRING)
class TFLayoutLMForTokenClassification(TFLayoutLMPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'mlm___cls', 'nsp___cls', 'cls.predictions', 'cls.seq_relationship']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: LayoutLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.layoutlm = TFLayoutLMMainLayer(config, add_pooling_layer=True, name='layoutlm')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.classifier = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, bbox: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFLayoutLMForTokenClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = TFLayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="tf")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = tf.convert_to_tensor([token_boxes])
        >>> token_labels = tf.convert_to_tensor([1, 1, 0, 0])

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=token_labels,
        ... )

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        logits = self.classifier(inputs=sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layoutlm', None) is not None:
            with tf.name_scope(self.layoutlm.name):
                self.layoutlm.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])