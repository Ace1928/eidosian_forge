from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
@add_start_docstrings('MobileBert Model with a `next sentence prediction (classification)` head on top.', MOBILEBERT_START_DOCSTRING)
class TFMobileBertForNextSentencePrediction(TFMobileBertPreTrainedModel, TFNextSentencePredictionLoss):
    _keys_to_ignore_on_load_unexpected = ['predictions___cls', 'cls.predictions']

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mobilebert = TFMobileBertMainLayer(config, name='mobilebert')
        self.cls = TFMobileBertOnlyNSPHead(config, name='seq_relationship___cls')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, next_sentence_label: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFNextSentencePredictorOutput]:
        """
        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        ```"""
        outputs = self.mobilebert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)
        next_sentence_loss = None if next_sentence_label is None else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (next_sentence_loss,) + output if next_sentence_loss is not None else output
        return TFNextSentencePredictorOutput(loss=next_sentence_loss, logits=seq_relationship_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'mobilebert', None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, 'cls', None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)