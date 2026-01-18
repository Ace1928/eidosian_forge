import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
@add_start_docstrings('The bare VitDet Transformer model outputting raw hidden-states without any specific head on top.', VITDET_START_DOCSTRING)
class VitDetModel(VitDetPreTrainedModel):

    def __init__(self, config: VitDetConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = VitDetEmbeddings(config)
        self.encoder = VitDetEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> VitDetEmbeddings:
        return self.embeddings.projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VITDET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import VitDetConfig, VitDetModel
        >>> import torch

        >>> config = VitDetConfig()
        >>> model = VitDetModel(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 768, 14, 14]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)