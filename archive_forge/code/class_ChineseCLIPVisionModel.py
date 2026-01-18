import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_chinese_clip import ChineseCLIPConfig, ChineseCLIPTextConfig, ChineseCLIPVisionConfig
@add_start_docstrings('The vision model from CHINESE_CLIP without any head or projection on top.', CHINESE_CLIP_START_DOCSTRING)
class ChineseCLIPVisionModel(ChineseCLIPPreTrainedModel):
    config_class = ChineseCLIPVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__(config)
        self.vision_model = ChineseCLIPVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ChineseCLIPVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, ChineseCLIPVisionModel

        >>> model = ChineseCLIPVisionModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)