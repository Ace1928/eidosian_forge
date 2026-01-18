import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
class LxmertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.visn_fc = LxmertVisualFeatureEncoder(config)
        self.config = config
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.layer = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_r_layers)])

    def forward(self, lang_feats, lang_attention_mask, visual_feats, visual_pos, visual_attention_mask=None, output_attentions=None):
        vision_hidden_states = ()
        language_hidden_states = ()
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None
        visual_feats = self.visn_fc(visual_feats, visual_pos)
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
            lang_feats = l_outputs[0]
            language_hidden_states = language_hidden_states + (lang_feats,)
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)
        for layer_module in self.r_layers:
            v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions=output_attentions)
            visual_feats = v_outputs[0]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)
        for layer_module in self.x_layers:
            x_outputs = layer_module(lang_feats, lang_attention_mask, visual_feats, visual_attention_mask, output_attentions=output_attentions)
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
        visual_encoder_outputs = (vision_hidden_states, vision_attentions if output_attentions else None)
        lang_encoder_outputs = (language_hidden_states, language_attentions if output_attentions else None)
        return (visual_encoder_outputs, lang_encoder_outputs, cross_encoder_attentions if output_attentions else None)