import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vilt import ViltConfig
class ViltEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.text_embeddings = TextEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViltPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def visual_embed(self, pixel_values, pixel_mask, max_image_length=200):
        _, _, ph, pw = self.patch_embeddings.projection.weight.shape
        x = self.patch_embeddings(pixel_values)
        x_mask = pixel_mask[:, None, :, :].float()
        x_mask = nn.functional.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        batch_size, num_channels, height, width = x.shape
        patch_dim = self.config.image_size // self.config.patch_size
        spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)
        pos_embed = torch.cat([nn.functional.pad(nn.functional.interpolate(spatial_pos, size=(h, w), mode='bilinear', align_corners=True), (0, width - w, 0, height - h)) for h, w in zip(x_h, x_w)], dim=0)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = torch.stack(meshgrid(torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1]), indexing='ij'), dim=-1).to(device=x_mask.device)
        patch_index = patch_index[None, None, :, :, :]
        patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
        patch_index = patch_index.flatten(1, 3)
        x_mask = x_mask.flatten(1)
        if max_image_length < 0 or max_image_length is None or (not isinstance(max_image_length, int)):
            effective_resolution = x_h * x_w
            max_image_length = effective_resolution.max()
        else:
            effective_resolution = x_h * x_w
            max_image_length = min(effective_resolution.max(), max_image_length)
        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]
        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_length - v for v in valid_nums]
        select = []
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
                select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))
        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat((self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed), dim=1)
        x = x + pos_embed
        x = self.dropout(x)
        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)
        return (x, x_mask, (patch_index, (height, width)))

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, inputs_embeds, image_embeds, image_token_type_idx=1):
        text_embeds = self.text_embeddings(input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        if image_embeds is None:
            image_embeds, image_masks, patch_index = self.visual_embed(pixel_values, pixel_mask, max_image_length=self.config.max_image_length)
        else:
            image_masks = pixel_mask.flatten(1)
        if image_token_type_idx is None:
            image_token_type_idx = 1
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device))
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device))
        embeddings = torch.cat([text_embeds, image_embeds], dim=1)
        masks = torch.cat([attention_mask, image_masks], dim=1)
        return (embeddings, masks)