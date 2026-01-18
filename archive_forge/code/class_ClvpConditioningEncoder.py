import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
class ClvpConditioningEncoder(nn.Module):
    """
    This class processes the log-mel spectrograms(extracted by the Feature Extractor) and text tokens(produced by the
    tokenizer) as inputs for the decoder model.

    First each log-mel spectrogram is processed into a single vector which captures valuable characteristics from each
    of them, then the text tokens are converted into token embeddings and position embeddings are added afterwards.
    Both of these vectors are concatenated and then passed to the decoder model.

    The text tokens helps to incorporate the "text information" and the log-mel spectrogram is used to specify the
    "voice characteristics" into the generated mel tokens.
    """

    def __init__(self, config: ClvpConfig):
        super().__init__()
        self.text_config = config.text_config
        self.decoder_config = config.decoder_config
        self.text_token_embedding = nn.Embedding(self.text_config.vocab_size, self.decoder_config.hidden_size)
        self.text_position_embedding = nn.Embedding(self.decoder_config.max_text_tokens, self.decoder_config.hidden_size)
        self.mel_conv = nn.Conv1d(self.decoder_config.feature_size, self.decoder_config.hidden_size, kernel_size=1)
        num_groups = self.compute_groupnorm_groups(self.decoder_config.hidden_size)
        self.group_norms = nn.ModuleList([nn.GroupNorm(num_groups, self.decoder_config.hidden_size, eps=1e-05, affine=True) for _ in range(self.decoder_config.num_mel_attn_blocks)])
        self.mel_attn_blocks = nn.ModuleList([ClvpSelfAttention(self.decoder_config) for _ in range(self.decoder_config.num_mel_attn_blocks)])
        self.gradient_checkpointing = False

    def compute_groupnorm_groups(self, channels: int, groups: int=32):
        """
        Calculates the value of `num_groups` for nn.GroupNorm. This logic is taken from the official tortoise
        repository. link :
        https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/models/arch_util.py#L26
        """
        if channels <= 16:
            groups = 8
        elif channels <= 64:
            groups = 16
        while channels % groups != 0:
            groups = int(groups / 2)
        if groups <= 2:
            raise ValueError(f'Number of groups for the GroupNorm must be greater than 2, but it is {groups}.Please consider using a different `hidden_size`')
        return groups

    def forward(self, input_features: torch.FloatTensor, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.size()
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = torch.ones([batch_size, seq_length], dtype=torch.long, device=input_ids.device)
        input_ids, attention_mask = _pad_extra_bos_eos_tokens(input_ids, attention_mask, bos_token_id=self.text_config.bos_token_id, eos_token_id=self.text_config.eos_token_id)
        inputs_embeds = self.text_token_embedding(input_ids)
        position_ids = attention_mask.cumsum(-1) - 1
        position_embeds = self.text_position_embedding(position_ids)
        text_embeds = inputs_embeds + position_embeds
        if self.gradient_checkpointing and self.training:
            mel_spec = torch.utils.checkpoint.checkpoint(self.mel_conv, input_features)
            for i, mel_attn_block in enumerate(self.mel_attn_blocks):
                residual_mel_spec = mel_spec.transpose(1, 2)
                mel_spec = torch.utils.checkpoint.checkpoint(self.group_norms[i], mel_spec).transpose(1, 2)
                mel_spec = torch.utils.checkpoint.checkpoint(mel_attn_block, mel_spec)[0] + residual_mel_spec
                mel_spec = mel_spec.transpose(1, 2)
        else:
            mel_spec = self.mel_conv(input_features)
            for i, mel_attn_block in enumerate(self.mel_attn_blocks):
                residual_mel_spec = mel_spec.transpose(1, 2)
                mel_spec = self.group_norms[i](mel_spec).transpose(1, 2)
                mel_spec = mel_attn_block(mel_spec)[0] + residual_mel_spec
                mel_spec = mel_spec.transpose(1, 2)
        mel_spec = mel_spec[:, :, 0]
        mel_spec = mel_spec.unsqueeze(1)
        if text_embeds.shape[0] == 1 and mel_spec.shape[0] != 1:
            text_embeds = text_embeds.repeat(mel_spec.shape[0], 1, 1)
        elif text_embeds.shape[0] != 1 and mel_spec.shape[0] == 1:
            mel_spec = mel_spec.repeat(text_embeds.shape[0], 1, 1)
        elif text_embeds.shape[0] != mel_spec.shape[0]:
            raise ValueError(f'The number of texts and number of audios must be same. Found {text_embeds.shape[0]} texts vs {mel_spec.shape[0]} audios')
        return torch.concat([mel_spec, text_embeds], dim=1)