import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
@add_start_docstrings('The bare Pix2StructVision Model transformer outputting raw hidden-states without any specific head on top.', PIX2STRUCT_VISION_START_DOCSTRING)
class Pix2StructVisionModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructVisionConfig
    main_input_name = 'flattened_patches'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Pix2StructVisionLayer']

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = Pix2StructVisionEmbeddings(config)
        self.encoder = Pix2StructVisionEncoder(config)
        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PIX2STRUCT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, flattened_patches: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:

        Example:

        ```python
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, Pix2StructVisionModel

        >>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 2048, 768]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if flattened_patches is None:
            raise ValueError('You have to specify flattened_patches')
        if attention_mask is None:
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(flattened_patches)
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)