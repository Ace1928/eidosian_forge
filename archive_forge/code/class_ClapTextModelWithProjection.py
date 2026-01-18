import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
@add_start_docstrings('\n    CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).\n    ', CLAP_START_DOCSTRING)
class ClapTextModelWithProjection(ClapPreTrainedModel):
    config_class = ClapTextConfig

    def __init__(self, config: ClapTextConfig):
        super().__init__(config)
        self.text_model = ClapTextModel(config)
        self.text_projection = ClapProjectionLayer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapTextModelOutput, config_class=ClapTextConfig)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ClapTextModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)
        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return ClapTextModelOutput(text_embeds=text_embeds, last_hidden_state=text_outputs.last_hidden_state, hidden_states=text_outputs.hidden_states, attentions=text_outputs.attentions)