import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vit_msn import ViTMSNConfig
@add_start_docstrings('The bare ViTMSN Model outputting raw hidden-states without any specific head on top.', VIT_MSN_START_DOCSTRING)
class ViTMSNModel(ViTMSNPreTrainedModel):

    def __init__(self, config: ViTMSNConfig, use_mask_token: bool=False):
        super().__init__(config)
        self.config = config
        self.embeddings = ViTMSNEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTMSNEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> ViTMSNPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_MSN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, bool_masked_pos: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, interpolate_pos_encoding: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, BaseModelOutput]:
        """
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMSNModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        >>> model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)