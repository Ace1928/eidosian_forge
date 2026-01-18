import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
@add_start_docstrings('The bare MGP-STR Model transformer outputting raw hidden-states without any specific head on top.', MGP_STR_START_DOCSTRING)
class MgpstrModel(MgpstrPreTrainedModel):

    def __init__(self, config: MgpstrConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = MgpstrEmbeddings(config)
        self.encoder = MgpstrEncoder(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.proj

    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return encoder_outputs
        return BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)