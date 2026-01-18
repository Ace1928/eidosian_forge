from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (AutoModel,
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from uform.torch_models import VisualEncoder
class ImageFeaturesPooler(nn.Module):

    def __init__(self, input_size, hidden_size, num_attn_heads, intermediate_size, num_latents, initializer_range):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        self.pooler = nn.TransformerDecoderLayer(hidden_size, num_attn_heads, intermediate_size, activation=nn.functional.silu, batch_first=True, norm_first=True)
        self.image_latents = nn.Parameter(torch.randn(1, num_latents, hidden_size) * initializer_range ** 0.5)

    def forward(self, features):
        features = self.projection(features)
        return self.pooler(self.image_latents.expand(features.shape[0], -1, -1), features)