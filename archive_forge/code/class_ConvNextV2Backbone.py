from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_convnextv2 import ConvNextV2Config
@add_start_docstrings('\n    ConvNeXT V2 backbone, to be used with frameworks like DETR and MaskFormer.\n    ', CONVNEXTV2_START_DOCSTRING)
class ConvNextV2Backbone(ConvNextV2PreTrainedModel, BackboneMixin):

    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)
        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm(num_channels, data_format='channels_first')
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnextv2-tiny-1k-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        embedding_output = self.embeddings(pixel_values)
        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=return_dict)
        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps += (hidden_state,)
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (hidden_states,)
            return output
        return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states if output_hidden_states else None, attentions=None)