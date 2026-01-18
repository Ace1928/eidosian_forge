import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
@add_start_docstrings('\n    KOSMOS-2 Model for generating text and image features. The model consists of a vision encoder and a language model.\n    ', KOSMOS2_START_DOCSTRING)
class Kosmos2Model(Kosmos2PreTrainedModel):
    config_class = Kosmos2Config
    main_input_name = 'pixel_values'

    def __init__(self, config: Kosmos2Config):
        super().__init__(config)
        self.text_model = Kosmos2TextModel(config.text_config)
        self.vision_model = Kosmos2VisionModel(config.vision_config)
        self.image_to_text_projection = Kosmos2ImageToTextProjection(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(KOSMOS2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2ModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, input_ids: Optional[torch.Tensor]=None, image_embeds_position_mask: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, image_embeds: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Kosmos2ModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Kosmos2Model

        >>> model = Kosmos2Model.from_pretrained("microsoft/kosmos-2-patch14-224")
        >>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        >>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = (
        ...     "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863>"
        ...     "</object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911>"
        ...     "</object>"
        ... )

        >>> inputs = processor(text=text, images=image, return_tensors="pt", add_eos_token=True)

        >>> last_hidden_state = model(
        ...     pixel_values=inputs["pixel_values"],
        ...     input_ids=inputs["input_ids"],
        ...     attention_mask=inputs["attention_mask"],
        ...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
        ... ).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 91, 2048]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_model_output = None
        projection_attentions = None
        if image_embeds is None:
            if pixel_values is None:
                raise ValueError('You have to specify either `pixel_values` or `image_embeds`.')
            vision_model_output = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            image_embeds = self.vision_model.model.post_layernorm(vision_model_output[0])
            image_embeds = nn.functional.normalize(image_embeds, dim=-1)
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, image_embeds=image_embeds, image_embeds_position_mask=image_embeds_position_mask, head_mask=head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            outputs = outputs + (image_embeds, projection_attentions, vision_model_output)
            return tuple((output for output in outputs if output is not None))
        return Kosmos2ModelOutput(last_hidden_state=outputs.last_hidden_state, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, image_embeds=image_embeds, projection_attentions=projection_attentions, vision_model_output=vision_model_output)