from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config
@add_start_docstrings('The bare MobileNetV1 model outputting raw hidden-states without any specific head on top.', MOBILENET_V1_START_DOCSTRING)
class MobileNetV1Model(MobileNetV1PreTrainedModel):

    def __init__(self, config: MobileNetV1Config, add_pooling_layer: bool=True):
        super().__init__(config)
        self.config = config
        depth = 32
        out_channels = max(int(depth * config.depth_multiplier), config.min_depth)
        self.conv_stem = MobileNetV1ConvLayer(config, in_channels=config.num_channels, out_channels=out_channels, kernel_size=3, stride=2)
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        self.layer = nn.ModuleList()
        for i in range(13):
            in_channels = out_channels
            if strides[i] == 2 or i == 0:
                depth *= 2
                out_channels = max(int(depth * config.depth_multiplier), config.min_depth)
            self.layer.append(MobileNetV1ConvLayer(config, in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=strides[i], groups=in_channels))
            self.layer.append(MobileNetV1ConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @add_start_docstrings_to_model_forward(MOBILENET_V1_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.conv_stem(pixel_values)
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        last_hidden_state = hidden_states
        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None
        if not return_dict:
            return tuple((v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None))
        return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=all_hidden_states)