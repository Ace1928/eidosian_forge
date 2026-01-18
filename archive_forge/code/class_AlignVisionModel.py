import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
@add_start_docstrings('The vision model from ALIGN without any head or projection on top.', ALIGN_START_DOCSTRING)
class AlignVisionModel(AlignPreTrainedModel):
    config_class = AlignVisionConfig
    main_input_name = 'pixel_values'
    supports_gradient_checkpointing = False

    def __init__(self, config: AlignVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = AlignVisionEmbeddings(config)
        self.encoder = AlignVisionEncoder(config)
        if config.pooling_type == 'mean':
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == 'max':
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.convolution

    @add_start_docstrings_to_model_forward(ALIGN_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=AlignVisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AlignVisionModel

        >>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")
        >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states)