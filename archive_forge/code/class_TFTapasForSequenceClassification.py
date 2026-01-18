from __future__ import annotations
import enum
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
from .configuration_tapas import TapasConfig
@add_start_docstrings('\n    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table\n    entailment tasks, such as TabFact (Chen et al., 2020).\n    ', TAPAS_START_DOCSTRING)
class TFTapasForSequenceClassification(TFTapasPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.tapas = TFTapasMainLayer(config, name='tapas')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')
        self.classifier = keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
            "classification_class_index" in the original implementation.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForSequenceClassification
        >>> import tensorflow as tf
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
        >>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = [
        ...     "There is only one actor who is 45 years old",
        ...     "There are 3 actors which played in more than 60 movies",
        ... ]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> labels = tf.convert_to_tensor([1, 0])  # 1 means entailed, 0 means refuted

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        outputs = self.tapas(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'tapas', None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])