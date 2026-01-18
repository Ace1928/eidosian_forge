from __future__ import annotations
import math
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
from .configuration_electra import ElectraConfig
@add_start_docstrings('\n    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.\n\n    Even though both the discriminator and generator may be loaded into this model, the discriminator is the only model\n    of the two to have the correct classification head to be used for this model.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForPreTraining(TFElectraPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.electra = TFElectraMainLayer(config, name='electra')
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name='discriminator_predictions')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFElectraForPreTrainingOutput, Tuple[tf.Tensor]]:
        """
        Returns:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFElectraForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        >>> model = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        >>> outputs = model(input_ids)
        >>> scores = outputs[0]
        ```"""
        discriminator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]
        return TFElectraForPreTrainingOutput(logits=logits, hidden_states=discriminator_hidden_states.hidden_states, attentions=discriminator_hidden_states.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'electra', None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        if getattr(self, 'discriminator_predictions', None) is not None:
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)