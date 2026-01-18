import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
@add_start_docstrings('The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on top.', BRIDGETOWER_START_DOCSTRING)
class BridgeTowerModel(BridgeTowerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        vision_config = config.vision_config
        text_config = config.text_config
        if config.share_cross_modal_transformer_layers:
            self.cross_modal_text_transform = nn.Linear(text_config.hidden_size, config.hidden_size)
            self.cross_modal_image_transform = nn.Linear(vision_config.hidden_size, config.hidden_size)
        else:
            self.cross_modal_text_transform = nn.ModuleList([nn.Linear(text_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)])
            self.cross_modal_image_transform = nn.ModuleList([nn.Linear(vision_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)])
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.vision_model = BridgeTowerVisionModel(vision_config)
        self.text_model = BridgeTowerTextModel(text_config)
        if not vision_config.share_layernorm and config.init_layernorm_from_vision_encoder:
            for ln in self.vision_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vision_model.visual.ln_post.weight.data
                ln.bias.data = self.vision_model.visual.ln_post.bias.data
        self.cross_modal_image_layers = nn.ModuleList([BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)])
        self.cross_modal_text_layers = nn.ModuleList([BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)])
        self.cross_modal_image_pooler = BridgeTowerPooler(config)
        self.cross_modal_text_pooler = BridgeTowerPooler(config)
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_modal_image_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.share_link_tower_layers:
            self.cross_modal_text_link_tower = BridgeTowerLinkTower(config)
            self.cross_modal_image_link_tower = BridgeTowerLinkTower(config)
        else:
            self.cross_modal_text_link_tower = nn.ModuleList([BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)])
            self.cross_modal_image_link_tower = nn.ModuleList([BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)])
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, image_token_type_idx: Optional[int]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], BridgeTowerModelOutput]:
        """
        output_hidden_states (`bool`, *optional*):
            If set to `True`, hidden states are returned as a list containing the hidden states of text, image, and
            cross-modal components respectively. i.e. `(hidden_states_text, hidden_states_image,
            hidden_states_cross_modal)` where each element is a list of the hidden states of the corresponding
            modality. `hidden_states_txt/img` are a list of tensors corresponding to unimodal hidden states and
            `hidden_states_cross_modal` is a list of tuples containing `cross_modal_text_hidden_states` and
            `cross_modal_image_hidden_states` of each brdige layer.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"
        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
        >>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.keys()
        odict_keys(['text_features', 'image_features', 'pooler_output'])
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        all_hidden_states_text = () if output_hidden_states else None
        all_hidden_states_image = () if output_hidden_states else None
        all_hidden_states_cross = () if output_hidden_states else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_token_type_idx = image_token_type_idx if image_token_type_idx else 1
        input_shape = input_ids.size()
        text_embeds = self.text_model.embeddings(input_ids=input_ids)
        if output_hidden_states:
            all_hidden_states_text += (text_embeds,)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=input_ids.device)
        extend_text_masks = self.text_model.get_extended_attention_mask(attention_mask, input_shape).to(input_ids.device)
        split_index = len(self.text_model.encoder.layer) - self.config.num_hidden_layers + 1
        for layer in self.text_model.encoder.layer[:split_index]:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
            if output_hidden_states:
                all_hidden_states_text += (text_embeds,)
        if image_embeds is None:
            image_embeds = self.vision_model.visual.forward_pre(pixel_values.type(self.vision_model.dtype))
        else:
            image_embeds = image_embeds.permute(1, 0, 2)
        if output_hidden_states:
            all_hidden_states_image += (image_embeds,)
        for block in self.vision_model.visual.transformer.resblocks[:split_index]:
            image_embeds = block(image_embeds)
            if output_hidden_states:
                all_hidden_states_image += (image_embeds,)
        image_embeds_with_ln = self.vision_model.visual.forward_post(image_embeds.type(self.vision_model.dtype))
        cross_modal_text = self.cross_modal_text_transform(text_embeds)
        text_token_type_embeddings = self.token_type_embeddings(torch.zeros(1, dtype=torch.long, device=input_ids.device)).expand_as(cross_modal_text)
        cross_modal_text = self.cross_modal_text_layernorm(cross_modal_text + text_token_type_embeddings)
        image_embeds_with_ln = self.cross_modal_image_transform(image_embeds_with_ln)
        image_token_type_embeddings = self.token_type_embeddings(torch.full((1,), image_token_type_idx, dtype=torch.long, device=input_ids.device)).expand_as(image_embeds_with_ln)
        image_embeds_with_ln = image_embeds_with_ln + image_token_type_embeddings
        cross_modal_image = self.cross_modal_image_layernorm(image_embeds_with_ln)
        pixel_mask = torch.ones((cross_modal_image.size(0), cross_modal_image.size(1)), dtype=torch.long, device=input_ids.device)
        extend_image_masks = self.text_model.get_extended_attention_mask(pixel_mask, pixel_mask.size()).to(input_ids.device)
        layer_outputs_text = self.cross_modal_text_layers[0](cross_modal_text, cross_modal_image, attention_mask=extend_text_masks, encoder_attention_mask=extend_image_masks, output_attentions=output_attentions)
        cross_text_features = layer_outputs_text[0]
        layer_outputs_image = self.cross_modal_image_layers[0](cross_modal_image, cross_modal_text, attention_mask=extend_image_masks, encoder_attention_mask=extend_text_masks, output_attentions=output_attentions)
        cross_image_features = layer_outputs_image[0]
        if output_hidden_states:
            all_hidden_states_cross += ((cross_text_features, cross_image_features),)
        if output_attentions:
            all_self_attentions += ((layer_outputs_text[1], layer_outputs_image[1]),)
        link_layer_index = 0
        for i in range(split_index, len(self.text_model.encoder.layer)):
            text_embeds = self.text_model.encoder.layer[i](text_embeds, extend_text_masks)[0]
            image_embeds = self.vision_model.visual.transformer.resblocks[i](image_embeds).type(self.vision_model.dtype)
            image_embeds_with_ln = self.cross_modal_image_transform(self.vision_model.visual.forward_post(image_embeds)) + image_token_type_embeddings
            text_link_tower = self.cross_modal_text_link_tower[link_layer_index]
            image_link_tower = self.cross_modal_image_link_tower[link_layer_index]
            cross_text_features_ = text_link_tower(self.cross_modal_text_transform(text_embeds) + text_token_type_embeddings, cross_text_features, extend_text_masks)
            cross_image_features_ = image_link_tower(image_embeds_with_ln, cross_image_features, extend_image_masks)
            layer_outputs_text = self.cross_modal_text_layers[link_layer_index + 1](cross_text_features_, cross_image_features_, attention_mask=extend_text_masks, encoder_attention_mask=extend_image_masks, output_attentions=output_attentions)
            cross_text_features = layer_outputs_text[0]
            layer_outputs_image = self.cross_modal_image_layers[link_layer_index + 1](cross_image_features_, cross_text_features_, attention_mask=extend_image_masks, encoder_attention_mask=extend_text_masks, output_attentions=output_attentions)
            cross_image_features = layer_outputs_image[0]
            link_layer_index += 1
            if output_hidden_states:
                all_hidden_states_text += (text_embeds,)
                all_hidden_states_image += (image_embeds,)
                all_hidden_states_cross += ((cross_text_features, cross_image_features),)
            if output_attentions:
                all_self_attentions += ((layer_outputs_text[1], layer_outputs_image[1]),)
        text_features, image_features = (cross_text_features, cross_image_features)
        cls_features = self.get_cls_features(text_features, image_features)
        if output_hidden_states:
            all_hidden_states = (all_hidden_states_text, all_hidden_states_image, all_hidden_states_cross)
        if not return_dict:
            return tuple((v for v in [text_features, image_features, cls_features, all_hidden_states, all_self_attentions] if v is not None))
        return BridgeTowerModelOutput(text_features=text_features, image_features=image_features, pooler_output=cls_features, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def get_cls_features(self, text_features, image_features):
        cls_features_text = self.cross_modal_text_pooler(text_features)
        cls_features_image = self.cross_modal_image_pooler(image_features)
        return torch.cat([cls_features_text, cls_features_image], dim=-1)