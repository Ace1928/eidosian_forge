import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimesformerEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type
        self.attention_type = attention_type
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if attention_type != 'space_only':
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings, num_frames, patch_width = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if embeddings.size(1) != self.position_embeddings.size(1):
            position_embeddings = self.position_embeddings
            cls_pos_embed = position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = position_embeddings[0, 1:, :].unsqueeze(0).transpose(1, 2)
            patch_num = int(other_pos_embed.size(2) ** 0.5)
            patch_height = embeddings.size(1) // patch_width
            other_pos_embed = other_pos_embed.reshape(1, embeddings.size(2), patch_num, patch_num)
            new_pos_embed = nn.functional.interpolate(other_pos_embed, size=(patch_height, patch_width), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            embeddings = embeddings + new_pos_embed
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.pos_drop(embeddings)
        if self.attention_type != 'space_only':
            cls_tokens = embeddings[:batch_size, 0, :].unsqueeze(1)
            embeddings = embeddings[:, 1:]
            _, patch_height, patch_width = embeddings.shape
            embeddings = embeddings.reshape(batch_size, num_frames, patch_height, patch_width).permute(0, 2, 1, 3).reshape(batch_size * patch_height, num_frames, patch_width)
            if num_frames != self.time_embeddings.size(1):
                time_embeddings = self.time_embeddings.transpose(1, 2)
                new_time_embeddings = nn.functional.interpolate(time_embeddings, size=num_frames, mode='nearest')
                new_time_embeddings = new_time_embeddings.transpose(1, 2)
                embeddings = embeddings + new_time_embeddings
            else:
                embeddings = embeddings + self.time_embeddings
            embeddings = self.time_drop(embeddings)
            embeddings = embeddings.view(batch_size, patch_height, num_frames, patch_width).reshape(batch_size, patch_height * num_frames, patch_width)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        return embeddings