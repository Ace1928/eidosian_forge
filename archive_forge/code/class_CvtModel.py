import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig
@add_start_docstrings('The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.', CVT_START_DOCSTRING)
class CvtModel(CvtPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CvtEncoder(config)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithCLSToken, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithCLSToken]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        encoder_outputs = self.encoder(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithCLSToken(last_hidden_state=sequence_output, cls_token_value=encoder_outputs.cls_token_value, hidden_states=encoder_outputs.hidden_states)