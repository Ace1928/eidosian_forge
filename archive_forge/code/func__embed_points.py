import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
    """Embeds point prompts."""
    points = points + 0.5
    if pad:
        target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
        target_labels_shape = (points.shape[0], points.shape[1], 1)
        padding_point = torch.zeros(target_point_shape, device=points.device)
        padding_label = -torch.ones(target_labels_shape, device=labels.device)
        points = torch.cat([points, padding_point], dim=2)
        labels = torch.cat([labels, padding_label], dim=2)
    input_shape = (self.input_image_size, self.input_image_size)
    point_embedding = self.shared_embedding(points, input_shape)
    point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)
    point_embedding = torch.where(labels[..., None] != -10, point_embedding, torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device))
    point_embedding = torch.where((labels == 0)[:, :, :, None], point_embedding + self.point_embed[0].weight[None, None, :, :], point_embedding)
    point_embedding = torch.where((labels == 1)[:, :, :, None], point_embedding + self.point_embed[1].weight[None, None, :, :], point_embedding)
    return point_embedding