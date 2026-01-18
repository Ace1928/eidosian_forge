import math
from typing import Dict, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilevit import MobileViTConfig
@add_start_docstrings('The bare MobileViT model outputting raw hidden-states without any specific head on top.', MOBILEVIT_START_DOCSTRING)
class MobileViTModel(MobileViTPreTrainedModel):

    def __init__(self, config: MobileViTConfig, expand_output: bool=True):
        super().__init__(config)
        self.config = config
        self.expand_output = expand_output
        self.conv_stem = MobileViTConvLayer(config, in_channels=config.num_channels, out_channels=config.neck_hidden_sizes[0], kernel_size=3, stride=2)
        self.encoder = MobileViTEncoder(config)
        if self.expand_output:
            self.conv_1x1_exp = MobileViTConvLayer(config, in_channels=config.neck_hidden_sizes[5], out_channels=config.neck_hidden_sizes[6], kernel_size=1)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer_index, heads in heads_to_prune.items():
            mobilevit_layer = self.encoder.layer[layer_index]
            if isinstance(mobilevit_layer, MobileViTLayer):
                for transformer_layer in mobilevit_layer.transformer.layer:
                    transformer_layer.attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.conv_stem(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None
        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            return output + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states)