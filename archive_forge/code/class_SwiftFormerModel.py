import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
@add_start_docstrings('The bare SwiftFormer Model transformer outputting raw hidden-states without any specific head on top.', SWIFTFORMER_START_DOCSTRING)
class SwiftFormerModel(SwiftFormerPreTrainedModel):

    def __init__(self, config: SwiftFormerConfig):
        super().__init__(config)
        self.config = config
        self.patch_embed = SwiftFormerPatchEmbedding(config)
        self.encoder = SwiftFormerEncoder(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(SWIFTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        """ """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.patch_embed(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return tuple((v for v in encoder_outputs if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=encoder_outputs.last_hidden_state, hidden_states=encoder_outputs.hidden_states)