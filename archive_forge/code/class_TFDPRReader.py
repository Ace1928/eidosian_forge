from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
@add_start_docstrings('The bare DPRReader transformer outputting span predictions.', TF_DPR_START_DOCSTRING)
class TFDPRReader(TFDPRPretrainedReader):

    def __init__(self, config: DPRConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.span_predictor = TFDPRSpanPredictorLayer(config, name='span_predictor')

    def get_input_embeddings(self):
        try:
            return self.span_predictor.encoder.bert_model.get_input_embeddings()
        except AttributeError:
            self.build()
            return self.span_predictor.encoder.bert_model.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRReaderOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, training: bool=False) -> TFDPRReaderOutput | Tuple[tf.Tensor, ...]:
        """
        Return:

        Examples:

        ```python
        >>> from transformers import TFDPRReader, DPRReaderTokenizer

        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> model = TFDPRReader.from_pretrained("facebook/dpr-reader-single-nq-base", from_pt=True)
        >>> encoded_inputs = tokenizer(
        ...     questions=["What is love ?"],
        ...     titles=["Haddaway"],
        ...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        ...     return_tensors="tf",
        ... )
        >>> outputs = model(encoded_inputs)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> relevance_logits = outputs.relevance_logits
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.ones(input_shape, dtype=tf.dtypes.int32)
        return self.span_predictor(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'span_predictor', None) is not None:
            with tf.name_scope(self.span_predictor.name):
                self.span_predictor.build(None)