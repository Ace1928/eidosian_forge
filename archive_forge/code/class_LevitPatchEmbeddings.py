import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitPatchEmbeddings(nn.Module):
    """
    LeViT patch embeddings, for final embeddings to be passed to transformer blocks. It consists of multiple
    `LevitConvEmbeddings`.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_layer_1 = LevitConvEmbeddings(config.num_channels, config.hidden_sizes[0] // 8, config.kernel_size, config.stride, config.padding)
        self.activation_layer_1 = nn.Hardswish()
        self.embedding_layer_2 = LevitConvEmbeddings(config.hidden_sizes[0] // 8, config.hidden_sizes[0] // 4, config.kernel_size, config.stride, config.padding)
        self.activation_layer_2 = nn.Hardswish()
        self.embedding_layer_3 = LevitConvEmbeddings(config.hidden_sizes[0] // 4, config.hidden_sizes[0] // 2, config.kernel_size, config.stride, config.padding)
        self.activation_layer_3 = nn.Hardswish()
        self.embedding_layer_4 = LevitConvEmbeddings(config.hidden_sizes[0] // 2, config.hidden_sizes[0], config.kernel_size, config.stride, config.padding)
        self.num_channels = config.num_channels

    def forward(self, pixel_values):
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embeddings = self.embedding_layer_1(pixel_values)
        embeddings = self.activation_layer_1(embeddings)
        embeddings = self.embedding_layer_2(embeddings)
        embeddings = self.activation_layer_2(embeddings)
        embeddings = self.embedding_layer_3(embeddings)
        embeddings = self.activation_layer_3(embeddings)
        embeddings = self.embedding_layer_4(embeddings)
        return embeddings.flatten(2).transpose(1, 2)