import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
def _calc_img_embeddings(self, image, bbox, position_ids):
    visual_embeddings = self.visual_proj(self.visual(image))
    position_embeddings = self.embeddings.position_embeddings(position_ids)
    spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
    embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
    if self.has_visual_segment_embedding:
        embeddings += self.visual_segment_embedding
    embeddings = self.visual_LayerNorm(embeddings)
    embeddings = self.visual_dropout(embeddings)
    return embeddings