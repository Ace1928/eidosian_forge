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
class TimesformerPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        embeddings = self.projection(pixel_values)
        patch_width = embeddings.size(-1)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return (embeddings, num_frames, patch_width)