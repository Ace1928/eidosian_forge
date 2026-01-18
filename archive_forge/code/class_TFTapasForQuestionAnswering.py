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
@add_start_docstrings('\n    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables\n    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for\n    SQA, WTQ or WikiSQL-supervised tasks.\n    ', TAPAS_START_DOCSTRING)
class TFTapasForQuestionAnswering(TFTapasPreTrainedModel):

    def __init__(self, config: TapasConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.tapas = TFTapasMainLayer(config, name='tapas')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.compute_token_logits = TFTapasComputeTokenLogits(config, name='compute_token_logits')
        self.compute_column_logits = TFTapasComputeColumnLogits(config, name='compute_column_logits')
        if config.num_aggregation_labels > 0:
            self.aggregation_classifier = keras.layers.Dense(config.num_aggregation_labels, kernel_initializer=get_initializer(config.initializer_range), name='aggregation_classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFTableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, table_mask: np.ndarray | tf.Tensor | None=None, aggregation_labels: np.ndarray | tf.Tensor | None=None, float_answer: np.ndarray | tf.Tensor | None=None, numeric_values: np.ndarray | tf.Tensor | None=None, numeric_values_scale: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFTableQuestionAnsweringOutput, Tuple[tf.Tensor]]:
        """
        table_mask (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and
            padding are 0.
        labels (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Labels per token for computing the hierarchical cell selection loss. This encodes the positions of the
            answer appearing in the table. Can be obtained using [`AutoTokenizer`].

            - 1 for tokens that are **part of the answer**,
            - 0 for tokens that are **not part of the answer**.

        aggregation_labels (`tf.Tensor` of shape `(batch_size, )`, *optional*):
            Aggregation function index for every example in the batch for computing the aggregation loss. Indices
            should be in `[0, ..., config.num_aggregation_labels - 1]`. Only required in case of strong supervision for
            aggregation (WikiSQL-supervised).
        float_answer (`tf.Tensor` of shape `(batch_size, )`, *optional*):
            Float answer for every example in the batch. Set to *float('nan')* for cell selection questions. Only
            required in case of weak supervision (WTQ) to calculate the aggregate mask and regression loss.
        numeric_values (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Numeric values of every token, NaN for tokens which are not numeric values. Can be obtained using
            [`AutoTokenizer`]. Only required in case of weak supervision for aggregation (WTQ) to calculate the
            regression loss.
        numeric_values_scale (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Scale of the numeric values of every token. Can be obtained using [`AutoTokenizer`]. Only required in case
            of weak supervision for aggregation (WTQ) to calculate the regression loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TapasForQuestionAnswering
        >>> import pandas as pd

        >>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        >>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

        >>> data = {
        ...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        ...     "Age": ["56", "45", "59"],
        ...     "Number of movies": ["87", "53", "69"],
        ... }
        >>> table = pd.DataFrame.from_dict(data)
        >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

        >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> logits_aggregation = outputs.logits_aggregation
        ```"""
        outputs = self.tapas(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape + [len(self.config.type_vocab_sizes)], 0)
        token_types = ['segment_ids', 'column_ids', 'row_ids', 'prev_labels', 'column_ranks', 'inv_column_ranks', 'numeric_relations']
        row_ids = token_type_ids[:, :, token_types.index('row_ids')]
        column_ids = token_type_ids[:, :, token_types.index('column_ids')]
        row_index = IndexMap(indices=tf.minimum(tf.cast(row_ids, tf.int32), self.config.max_num_rows - 1), num_segments=self.config.max_num_rows, batch_dims=1)
        col_index = IndexMap(indices=tf.minimum(tf.cast(column_ids, tf.int32), self.config.max_num_columns - 1), num_segments=self.config.max_num_columns, batch_dims=1)
        cell_index = ProductIndexMap(row_index, col_index)
        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)[:-1]
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)
        if table_mask is None:
            table_mask = tf.where(row_ids > 0, tf.ones_like(row_ids), tf.zeros_like(row_ids))
        input_mask_float = tf.cast(attention_mask, tf.float32)
        table_mask_float = tf.cast(table_mask, tf.float32)
        cell_mask, _ = reduce_mean(input_mask_float, cell_index)
        logits = self.compute_token_logits(sequence_output)
        column_logits = None
        if self.config.select_one_column:
            column_logits = self.compute_column_logits(sequence_output, cell_index, cell_mask, self.config.allow_empty_column_selection)
        logits_aggregation = None
        if self.config.num_aggregation_labels > 0:
            logits_aggregation = self.aggregation_classifier(pooled_output)
        total_loss = tf.zeros(shape=(1,), dtype=tf.float32)
        calculate_loss = False
        if labels is not None:
            calculate_loss = True
            is_supervised = not self.config.num_aggregation_labels > 0 or not self.config.use_answer_as_supervision
            if is_supervised:
                aggregate_mask = None
            elif float_answer is not None:
                assert shape_list(labels)[0] == shape_list(float_answer)[0], 'Make sure the answers are a FloatTensor of shape (batch_size,)'
                aggregate_mask = _calculate_aggregate_mask(float_answer, pooled_output, self.config.cell_selection_preference, labels, self.aggregation_classifier)
            else:
                aggregate_mask = None
                raise ValueError('You have to specify float answers in order to calculate the aggregate mask')
            if self.config.average_logits_per_cell:
                logits_per_cell, _ = reduce_mean(logits, cell_index)
                logits = gather(logits_per_cell, cell_index)
            dist_per_token = tfp.distributions.Bernoulli(logits=logits)
            selection_loss_per_example = None
            if not self.config.select_one_column:
                weight = tf.where(labels == 0, tf.ones_like(labels, dtype=tf.float32), self.config.positive_label_weight * tf.ones_like(labels, dtype=tf.float32))
                selection_loss_per_token = -dist_per_token.log_prob(labels) * weight
                selection_loss_per_example = tf.reduce_sum(selection_loss_per_token * input_mask_float, axis=1) / (tf.reduce_sum(input_mask_float, axis=1) + EPSILON_ZERO_DIVISION)
            else:
                selection_loss_per_example, logits = _single_column_cell_selection_loss(logits, column_logits, labels, cell_index, col_index, cell_mask)
                dist_per_token = tfp.distributions.Bernoulli(logits=logits)
            if self.config.disable_per_token_loss:
                pass
            elif is_supervised:
                total_loss += tf.reduce_mean(selection_loss_per_example)
            else:
                total_loss += tf.reduce_mean(selection_loss_per_example * (1.0 - aggregate_mask))
            if self.config.num_aggregation_labels > 0:
                if is_supervised:
                    if aggregation_labels is not None:
                        assert shape_list(labels)[0] == shape_list(aggregation_labels)[0], 'Make sure the aggregation labels are a LongTensor of shape (batch_size,)'
                        per_example_additional_loss = _calculate_aggregation_loss(logits_aggregation, aggregate_mask, aggregation_labels, self.config.use_answer_as_supervision, self.config.num_aggregation_labels, self.config.aggregation_loss_weight)
                    else:
                        raise ValueError('You have to specify aggregation labels in order to calculate the aggregation loss')
                else:
                    aggregation_labels = tf.zeros(shape_list(labels)[0], dtype=tf.int32)
                    per_example_additional_loss = _calculate_aggregation_loss(logits_aggregation, aggregate_mask, aggregation_labels, self.config.use_answer_as_supervision, self.config.num_aggregation_labels, self.config.aggregation_loss_weight)
                if self.config.use_answer_as_supervision:
                    if numeric_values is not None and numeric_values_scale is not None:
                        assert shape_list(numeric_values) == shape_list(numeric_values_scale)
                        answer_loss, large_answer_loss_mask = _calculate_regression_loss(float_answer, aggregate_mask, dist_per_token, numeric_values, numeric_values_scale, table_mask_float, logits_aggregation, self.config)
                        per_example_additional_loss += answer_loss
                        per_example_additional_loss *= large_answer_loss_mask
                    else:
                        raise ValueError('You have to specify numeric values and numeric values scale in order to calculate the regression loss')
                total_loss += tf.reduce_mean(per_example_additional_loss)
        else:
            labels = tf.zeros_like(logits)
            _, logits = _single_column_cell_selection_loss(logits, column_logits, labels, cell_index, col_index, cell_mask)
        if not return_dict:
            output = (logits, logits_aggregation) + outputs[2:]
            return (total_loss,) + output if calculate_loss else output
        return TFTableQuestionAnsweringOutput(loss=total_loss if calculate_loss else None, logits=logits, logits_aggregation=logits_aggregation, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'tapas', None) is not None:
            with tf.name_scope(self.tapas.name):
                self.tapas.build(None)
        if getattr(self, 'compute_token_logits', None) is not None:
            with tf.name_scope(self.compute_token_logits.name):
                self.compute_token_logits.build(None)
        if getattr(self, 'compute_column_logits', None) is not None:
            with tf.name_scope(self.compute_column_logits.name):
                self.compute_column_logits.build(None)
        if getattr(self, 'aggregation_classifier', None) is not None:
            with tf.name_scope(self.aggregation_classifier.name):
                self.aggregation_classifier.build([None, None, self.config.hidden_size])