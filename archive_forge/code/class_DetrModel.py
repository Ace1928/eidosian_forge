import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_detr import DetrConfig
@add_start_docstrings('\n    The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without\n    any specific head on top.\n    ', DETR_START_DOCSTRING)
class DetrModel(DetrPreTrainedModel):

    def __init__(self, config: DetrConfig):
        super().__init__(config)
        backbone = DetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = DetrConvModel(backbone, object_queries)
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], DetrModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 100, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)
        features, object_queries_list = self.backbone(pixel_values, pixel_mask)
        feature_map, mask = features[-1]
        if mask is None:
            raise ValueError('Backbone does not return downsampled pixel mask')
        projected_feature_map = self.input_projection(feature_map)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=flattened_features, attention_mask=flattened_mask, object_queries=object_queries, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)
        decoder_outputs = self.decoder(inputs_embeds=queries, attention_mask=None, object_queries=object_queries, query_position_embeddings=query_position_embeddings, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=flattened_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return DetrModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, intermediate_hidden_states=decoder_outputs.intermediate_hidden_states)