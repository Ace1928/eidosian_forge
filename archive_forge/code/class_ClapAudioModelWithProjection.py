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
@add_start_docstrings('\n    CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).\n    ', CLAP_START_DOCSTRING)
class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config_class = ClapAudioConfig
    main_input_name = 'input_features'

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)
        self.audio_model = ClapAudioModel(config)
        self.audio_projection = ClapProjectionLayer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_model.audio_encoder.patch_embed.proj

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapAudioModelOutput, config_class=ClapAudioConfig)
    def forward(self, input_features: Optional[torch.FloatTensor]=None, is_longer: Optional[torch.BoolTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ClapAudioModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import ClapAudioModelWithProjection, ClapProcessor

        >>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        >>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> inputs = processor(audios=audio_sample, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> audio_embeds = outputs.audio_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        audio_outputs = self.audio_model(input_features=input_features, is_longer=is_longer, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(pooled_output)
        if not return_dict:
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return ClapAudioModelOutput(audio_embeds=audio_embeds, last_hidden_state=audio_outputs.last_hidden_state, attentions=audio_outputs.attentions, hidden_states=audio_outputs.hidden_states)