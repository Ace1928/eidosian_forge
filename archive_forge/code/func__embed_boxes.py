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
def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
    """Embeds box prompts."""
    boxes = boxes + 0.5
    batch_size, nb_boxes = boxes.shape[:2]
    coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
    input_shape = (self.input_image_size, self.input_image_size)
    corner_embedding = self.shared_embedding(coords, input_shape)
    corner_embedding[:, :, 0, :] += self.point_embed[2].weight
    corner_embedding[:, :, 1, :] += self.point_embed[3].weight
    return corner_embedding