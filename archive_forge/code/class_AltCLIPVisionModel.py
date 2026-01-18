import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_altclip import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig
class AltCLIPVisionModel(AltCLIPPreTrainedModel):
    config_class = AltCLIPVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__(config)
        self.vision_model = AltCLIPVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(ALTCLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=AltCLIPVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AltCLIPVisionModel

        >>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)