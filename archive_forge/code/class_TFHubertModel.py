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
@add_start_docstrings('The bare TFHubert Model transformer outputing raw hidden-states without any specific head on top.', HUBERT_START_DOCSTRING)
class TFHubertModel(TFHubertPreTrainedModel):

    def __init__(self, config: HubertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.hubert = TFHubertMainLayer(config, name='hubert')

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, TFHubertModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        return_dict = return_dict if return_dict else self.config.return_dict
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'hubert', None) is not None:
            with tf.name_scope(self.hubert.name):
                self.hubert.build(None)