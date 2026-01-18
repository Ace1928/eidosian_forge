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
class LxmertXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.visual_attention = LxmertCrossAttentionLayer(config)
        self.lang_self_att = LxmertSelfAttentionLayer(config)
        self.visn_self_att = LxmertSelfAttentionLayer(config)
        self.lang_inter = LxmertIntermediate(config)
        self.lang_output = LxmertOutput(config)
        self.visn_inter = LxmertIntermediate(config)
        self.visn_output = LxmertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask, output_x_attentions=False):
        lang_att_output = self.visual_attention(lang_input, visual_input, ctx_att_mask=visual_attention_mask, output_attentions=output_x_attentions)
        visual_att_output = self.visual_attention(visual_input, lang_input, ctx_att_mask=lang_attention_mask, output_attentions=False)
        return (lang_att_output, visual_att_output)

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return (lang_att_output[0], visual_att_output[0])

    def output_fc(self, lang_input, visual_input):
        lang_inter_output = self.lang_inter(lang_input)
        visual_inter_output = self.visn_inter(visual_input)
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visual_output = self.visn_output(visual_inter_output, visual_input)
        return (lang_output, visual_output)

    def forward(self, lang_feats, lang_attention_mask, visual_feats, visual_attention_mask, output_attentions=False):
        lang_att_output, visual_att_output = self.cross_att(lang_input=lang_feats, lang_attention_mask=lang_attention_mask, visual_input=visual_feats, visual_attention_mask=visual_attention_mask, output_x_attentions=output_attentions)
        attention_probs = lang_att_output[1:]
        lang_att_output, visual_att_output = self.self_att(lang_att_output[0], lang_attention_mask, visual_att_output[0], visual_attention_mask)
        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        return (lang_output, visual_output, attention_probs[0]) if output_attentions else (lang_output, visual_output)