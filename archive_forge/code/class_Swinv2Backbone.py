import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_swinv2 import Swinv2Config
@add_start_docstrings('\n    Swinv2 backbone, to be used with frameworks like DETR and MaskFormer.\n    ', SWINV2_START_DOCSTRING)
class Swinv2Backbone(Swinv2PreTrainedModel, BackboneMixin):

    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2 ** i) for i in range(len(config.depths))]
        self.embeddings = Swinv2Embeddings(config)
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> BackboneOutput:
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

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/swinv2-tiny-patch4-window8-256", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        embedding_output, input_dimensions = self.embeddings(pixel_values)
        outputs = self.encoder(embedding_output, input_dimensions, head_mask=None, output_attentions=output_attentions, output_hidden_states=True, output_hidden_states_before_downsampling=True, return_dict=return_dict)
        hidden_states = outputs.reshaped_hidden_states if return_dict else outputs[-1]
        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                feature_maps += (hidden_state,)
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs[1],)
            if output_attentions:
                output += (outputs[2],)
            return output
        return BackboneOutput(feature_maps=feature_maps, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)