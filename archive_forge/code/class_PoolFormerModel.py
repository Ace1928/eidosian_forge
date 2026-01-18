import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_poolformer import PoolFormerConfig
@add_start_docstrings('The bare PoolFormer Model transformer outputting raw hidden-states without any specific head on top.', POOLFORMER_START_DOCSTRING)
class PoolFormerModel(PoolFormerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = PoolFormerEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        encoder_outputs = self.encoder(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output, None) + encoder_outputs[1:]
        return BaseModelOutputWithNoAttention(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states)