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
class VLMPreTrainedModel(PreTrainedModel):
    config_class = VLMConfig
    base_model_prefix = 'vlm'
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        pass

    def _initialize_weights(self, module):
        pass